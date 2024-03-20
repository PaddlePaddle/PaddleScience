# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Tuple

import einops
import paddle
from paddle.nn import PixelShuffle
from paddle.nn import PixelUnshuffle

paddle.seed(42)


class DGMR(paddle.nn.Layer):
    """Deep Generative Model of Radar.
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954.
        but slightly modified for multiple satellite channels

    Args:
        forecast_steps (int, optional): Number of steps to predict in the future
        input_channels (int, optional): Number of input channels per image
        gen_lr (float, optional): Learning rate for the generator
        disc_lr (float, optional): Learning rate for the discriminators, shared for both temporal and spatial discriminator
        conv_type (str, optional): Type of 2d convolution to use, see satflow/models/utils.py for options
        beta1 (float, optional): Beta1 for Adam optimizer
        beta2 (float, optional): Beta2 for Adam optimizer
        num_samples (int, optional): Number of samples of the latent space to sample for training/validation
        grid_lambda (float, optional): Lambda for the grid regularization loss
        output_shape (int, optional): Shape of the output predictions, generally should be same as the input shape
        generation_steps (int, optional): Number of generation steps to use in forward pass, in paper is 6 and the best is chosen for the loss
            this results in huge amounts of GPU memory though, so less might work better for training.
        context_channels (int, optional): Number of output channels for the lowest block of conditioning stack
        latent_channels (int, optional): Number of channels that the latent space should be reshaped to,
            input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.DGMR()
    """

    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 5e-05,
        disc_lr: float = 0.0002,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768,
        context_channels: int = 384,
        generation_steps: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.generation_steps = generation_steps
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = Generator(
            self.conditioning_stack, self.latent_stack, self.sampler
        )
        self.discriminator = Discriminator(input_channels)
        self.global_iteration = 0
        self.automatic_optimization = False

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch
        images = images.astype(dtype="float32")
        future_images = future_images.astype(dtype="float32")
        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = self(images)
            generated_sequence = paddle.concat(x=[images, predictions], axis=1)
            real_sequence = paddle.concat(x=[images, future_images], axis=1)
            concatenated_inputs = paddle.concat(
                x=[real_sequence, generated_sequence], axis=0
            )
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = paddle.split(
                x=concatenated_outputs,
                num_or_sections=[real_sequence.shape[0], generated_sequence.shape[0]],
                axis=0,
            )
            score_real_spatial, score_real_temporal = paddle.split(
                x=score_real, num_or_sections=1, axis=1
            )
            score_generated_spatial, score_generated_temporal = paddle.split(
                x=score_generated, num_or_sections=1, axis=1
            )
            discriminator_loss = loss_hinge_disc(
                score_generated_spatial, score_real_spatial
            ) + loss_hinge_disc(score_generated_temporal, score_real_temporal)
            self.manual_backward(discriminator_loss)
            d_opt.step()
            d_opt.clear_grad()
        ######################
        # Optimize Generator #
        ######################
        predictions = [self(images) for _ in range(self.generation_steps)]
        grid_cell_reg = grid_cell_regularizer(
            paddle.stack(x=predictions, axis=0), future_images
        )
        generated_sequence = [paddle.concat(x=[images, x], axis=1) for x in predictions]
        real_sequence = paddle.concat(x=[images, future_images], axis=1)
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = paddle.concat(x=[real_sequence, g_seq], axis=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = paddle.split(
                x=concatenated_outputs,
                num_or_sections=[real_sequence.shape[0], g_seq.shape[0]],
                axis=0,
            )
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(paddle.concat(x=generated_scores, axis=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        self.manual_backward(generator_loss)
        g_opt.step()
        g_opt.clear_grad()

        return discriminator_loss, generator_loss, grid_cell_reg

    def validation_step(self, batch, batch_idx):
        images, future_images = batch
        images = images.astype(dtype="float32")
        future_images = future_images.astype(dtype="float32")
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = self(images)
            generated_sequence = paddle.concat(x=[images, predictions], axis=1)
            real_sequence = paddle.concat(x=[images, future_images], axis=1)
            concatenated_inputs = paddle.concat(
                x=[real_sequence, generated_sequence], axis=0
            )
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = paddle.split(
                x=concatenated_outputs,
                num_or_sections=[real_sequence.shape[0], generated_sequence.shape[0]],
                axis=0,
            )
            score_real_spatial, score_real_temporal = paddle.split(
                x=score_real, num_or_sections=score_real.shape[1], axis=1
            )
            score_generated_spatial, score_generated_temporal = paddle.split(
                x=score_generated, num_or_sections=score_generated.shape[1], axis=1
            )
            discriminator_loss = loss_hinge_disc(
                score_generated_spatial, score_real_spatial
            ) + loss_hinge_disc(score_generated_temporal, score_real_temporal)
        ######################
        # Optimize Generator #
        ######################
        predictions = [self(images) for _ in range(self.generation_steps)]
        grid_cell_reg = grid_cell_regularizer(
            paddle.stack(x=predictions, axis=0), future_images
        )
        generated_sequence = [paddle.concat(x=[images, x], axis=1) for x in predictions]
        real_sequence = paddle.concat(x=[images, future_images], axis=1)
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = paddle.concat(x=[real_sequence, g_seq], axis=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = paddle.split(
                x=concatenated_outputs,
                num_or_sections=[real_sequence.shape[0], g_seq.shape[0]],
                axis=0,
            )
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(paddle.concat(x=generated_scores, axis=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        return discriminator_loss, generator_loss, grid_cell_reg

    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2
        opt_g = paddle.optimizer.Adam(
            parameters=self.generator.parameters(),
            learning_rate=self.gen_lr,
            beta1=(b1, b2)[0],
            beta2=(b1, b2)[1],
            weight_decay=0.0,
        )
        opt_d = paddle.optimizer.Adam(
            parameters=self.discriminator.parameters(),
            learning_rate=self.disc_lr,
            beta1=(b1, b2)[0],
            beta2=(b1, b2)[1],
            weight_decay=0.0,
        )
        return [opt_g, opt_d], []


class Sampler(paddle.nn.Layer):
    def __init__(
        self,
        forecast_steps: int = 18,
        latent_channels: int = 768,
        context_channels: int = 384,
        output_channels: int = 1,
        **kwargs,
    ):
        """
        Sampler from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        The sampler takes the output from the Latent and Context conditioning stacks and
        creates one stack of ConvGRU layers per future timestep.

        Args:
            forecast_steps: Number of forecast steps
            latent_channels: Number of input channels to the lowest ConvGRU layer
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        self.forecast_steps = self.config["forecast_steps"]
        latent_channels = self.config["latent_channels"]
        context_channels = self.config["context_channels"]
        output_channels = self.config["output_channels"]
        self.convGRU1 = ConvGRU(
            input_channels=latent_channels + context_channels,
            output_channels=context_channels,
            kernel_size=3,
        )
        self.gru_conv_1x1 = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=context_channels,
                out_channels=latent_channels,
                kernel_size=(1, 1),
            )
        )
        self.g1 = GBlock(
            input_channels=latent_channels, output_channels=latent_channels
        )
        self.up_g1 = UpsampleGBlock(
            input_channels=latent_channels, output_channels=latent_channels // 2
        )
        self.convGRU2 = ConvGRU(
            input_channels=latent_channels // 2 + context_channels // 2,
            output_channels=context_channels // 2,
            kernel_size=3,
        )
        self.gru_conv_1x1_2 = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=context_channels // 2,
                out_channels=latent_channels // 2,
                kernel_size=(1, 1),
            )
        )
        self.g2 = GBlock(
            input_channels=latent_channels // 2, output_channels=latent_channels // 2
        )
        self.up_g2 = UpsampleGBlock(
            input_channels=latent_channels // 2, output_channels=latent_channels // 4
        )
        self.convGRU3 = ConvGRU(
            input_channels=latent_channels // 4 + context_channels // 4,
            output_channels=context_channels // 4,
            kernel_size=3,
        )
        self.gru_conv_1x1_3 = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=context_channels // 4,
                out_channels=latent_channels // 4,
                kernel_size=(1, 1),
            )
        )
        self.g3 = GBlock(
            input_channels=latent_channels // 4, output_channels=latent_channels // 4
        )
        self.up_g3 = UpsampleGBlock(
            input_channels=latent_channels // 4, output_channels=latent_channels // 8
        )
        self.convGRU4 = ConvGRU(
            input_channels=latent_channels // 8 + context_channels // 8,
            output_channels=context_channels // 8,
            kernel_size=3,
        )
        self.gru_conv_1x1_4 = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=context_channels // 8,
                out_channels=latent_channels // 8,
                kernel_size=(1, 1),
            )
        )
        self.g4 = GBlock(
            input_channels=latent_channels // 8, output_channels=latent_channels // 8
        )
        self.up_g4 = UpsampleGBlock(
            input_channels=latent_channels // 8, output_channels=latent_channels // 16
        )
        self.bn = paddle.nn.BatchNorm2D(num_features=latent_channels // 16)
        self.relu = paddle.nn.ReLU()
        self.conv_1x1 = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=latent_channels // 16,
                out_channels=4 * output_channels,
                kernel_size=(1, 1),
            )
        )
        self.depth2space = PixelShuffle(upscale_factor=2)

    def forward(
        self, conditioning_states: List[paddle.Tensor], latent_dim: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Perform the sampling from Skillful Nowcasting with GANs
        Args:
            conditioning_states: Outputs from the `ContextConditioningStack` with the 4 input states, ordered from largest to smallest spatially
            latent_dim: Output from `LatentConditioningStack` for input into the ConvGRUs

        Returns:
            forecast_steps-length output of images for future timesteps

        """
        init_states = conditioning_states
        latent_dim = einops.repeat(
            latent_dim, "b c h w -> (repeat b) c h w", repeat=init_states[0].shape[0]
        )
        hidden_states = [latent_dim] * self.forecast_steps

        hidden_states = self.convGRU1(hidden_states, init_states[3])
        hidden_states = [self.gru_conv_1x1(h) for h in hidden_states]
        hidden_states = [self.g1(h) for h in hidden_states]
        hidden_states = [self.up_g1(h) for h in hidden_states]
        hidden_states = self.convGRU2(hidden_states, init_states[2])
        hidden_states = [self.gru_conv_1x1_2(h) for h in hidden_states]
        hidden_states = [self.g2(h) for h in hidden_states]
        hidden_states = [self.up_g2(h) for h in hidden_states]
        hidden_states = self.convGRU3(hidden_states, init_states[1])
        hidden_states = [self.gru_conv_1x1_3(h) for h in hidden_states]
        hidden_states = [self.g3(h) for h in hidden_states]
        hidden_states = [self.up_g3(h) for h in hidden_states]
        hidden_states = self.convGRU4(hidden_states, init_states[0])
        hidden_states = [self.gru_conv_1x1_4(h) for h in hidden_states]
        hidden_states = [self.g4(h) for h in hidden_states]
        hidden_states = [self.up_g4(h) for h in hidden_states]
        hidden_states = [paddle.nn.functional.relu(x=self.bn(h)) for h in hidden_states]
        hidden_states = [self.conv_1x1(h) for h in hidden_states]
        hidden_states = [self.depth2space(h) for h in hidden_states]
        forecasts = paddle.stack(x=hidden_states, axis=1)
        return forecasts


class Generator(paddle.nn.Layer):
    def __init__(
        self,
        conditioning_stack: paddle.nn.Layer,
        latent_stack: paddle.nn.Layer,
        sampler: paddle.nn.Layer,
    ):
        """
        Wraps the three parts of the generator for simpler calling
        Args:
            conditioning_stack:
            latent_stack:
            sampler:
        """
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.latent_stack = latent_stack
        self.sampler = sampler

    def forward(self, x):
        conditioning_states = self.conditioning_stack(x)
        latent_dim = self.latent_stack(x)
        x = self.sampler(conditioning_states, latent_dim)
        return x


class Discriminator(paddle.nn.Layer):
    def __init__(
        self,
        input_channels: int = 12,
        num_spatial_frames: int = 8,
        conv_type: str = "standard",
        **kwargs,
    ):
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_spatial_frames = self.config["num_spatial_frames"]
        conv_type = self.config["conv_type"]
        self.spatial_discriminator = SpatialDiscriminator(
            input_channels=input_channels,
            num_timesteps=num_spatial_frames,
            conv_type=conv_type,
        )
        self.temporal_discriminator = TemporalDiscriminator(
            input_channels=input_channels, conv_type=conv_type
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        spatial_loss = self.spatial_discriminator(x)
        temporal_loss = self.temporal_discriminator(x)
        return paddle.concat(x=[spatial_loss, temporal_loss], axis=1)


class TemporalDiscriminator(paddle.nn.Layer):
    def __init__(
        self,
        input_channels: int = 12,
        num_layers: int = 3,
        conv_type: str = "standard",
        **kwargs,
    ):
        """
        Temporal Discriminator from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of channels per timestep
            crop_size: Size of the crop, in the paper half the width of the input images
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        self.downsample = paddle.nn.AvgPool3D(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), exclusive=False
        )
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 48
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=internal_chn * input_channels,
            conv_type="3d",
            first_relu=False,
        )
        self.d2 = DBlock(
            input_channels=internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            conv_type="3d",
        )
        self.intermediate_dblocks = paddle.nn.LayerList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d_last = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )
        self.fc = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Linear(
                in_features=2 * internal_chn * input_channels, out_features=1
            )
        )
        self.relu = paddle.nn.ReLU()
        self.bn = paddle.nn.BatchNorm1D(num_features=2 * internal_chn * input_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.downsample(x)
        if len(x.shape) == 4:
            x = self.space2depth(x)
        elif len(x.shape) == 5:
            x_list = []
            for i in range(x.shape[0]):
                x_list.append(self.space2depth(x[i]))
            x = paddle.stack(x_list, axis=0)
        x = paddle.transpose(x=x, perm=(0, 2, 1, 3, 4))
        x = self.d1(x)
        x = self.d2(x)
        x = paddle.transpose(x=x, perm=(0, 2, 1, 3, 4))
        representations = []
        for idx in range(x.shape[1]):
            rep = x[:, idx, :, :, :]
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d_last(rep)
            rep = paddle.sum(x=paddle.nn.functional.relu(x=rep), axis=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            representations.append(rep)
        x = paddle.stack(x=representations, axis=1)
        x = paddle.sum(x=x, keepdim=True, axis=1)
        return x


class SpatialDiscriminator(paddle.nn.Layer):
    def __init__(
        self,
        input_channels: int = 12,
        num_timesteps: int = 8,
        num_layers: int = 4,
        conv_type: str = "standard",
        **kwargs,
    ):
        """
        Spatial discriminator from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            num_timesteps: Number of timesteps to use, in the paper 8/18 timesteps were chosen
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_timesteps = self.config["num_timesteps"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        self.num_timesteps = num_timesteps
        self.mean_pool = paddle.nn.AvgPool2D(kernel_size=2, exclusive=False)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=2 * internal_chn * input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = paddle.nn.LayerList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )
        self.fc = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Linear(
                in_features=2 * internal_chn * input_channels, out_features=1
            )
        )
        self.relu = paddle.nn.ReLU()
        self.bn = paddle.nn.BatchNorm1D(num_features=2 * internal_chn * input_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        idxs = paddle.randint(low=0, high=x.shape[1], shape=(self.num_timesteps,))
        representations = []
        for idx in idxs:
            # for idx in range(x.shape[1]):
            rep = self.mean_pool(x[:, idx, :, :, :])
            if len(rep.shape) == 4:
                rep = self.space2depth(rep)
            elif len(rep.shape) == 5:
                rep_list = []
                for i in range(rep.shape[0]):
                    rep_list.append(self.space2depth(rep[i]))
                rep = paddle.stack(rep_list, axis=0)
            rep = self.d1(rep)
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d6(rep)
            rep = paddle.sum(x=paddle.nn.functional.relu(x=rep), axis=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            """
            Pseudocode from DeepMind
            # Sum-pool the representations and feed to spectrally normalized lin. layer.
            y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
            y = layers.BatchNorm(calc_sigma=False)(y)
            output_layer = layers.Linear(output_size=1)
            output = output_layer(y)

            # Take the sum across the t samples. Note: we apply the ReLU to
            # (1 - score_real) and (1 + score_generated) in the loss.
            output = tf.reshape(output, [b, n, 1])
            output = tf.reduce_sum(output, keepdims=True, axis=1)
            return output
            """
            representations.append(rep)
        x = paddle.stack(x=representations, axis=1)
        x = paddle.sum(x=x, keepdim=True, axis=1)
        return x


class GBlock(paddle.nn.Layer):
    """Residual generator block without upsampling"""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        spectral_normalized_eps=0.0001,
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = paddle.nn.BatchNorm2D(num_features=input_channels)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=input_channels)
        self.relu = paddle.nn.ReLU()
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=1
            ),
            eps=spectral_normalized_eps,
        )
        self.first_conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.shape[1] != self.output_channels:
            sc = self.conv_1x1(x)
        else:
            sc = x
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        x = x2 + sc
        return x


class UpsampleGBlock(paddle.nn.Layer):
    """Residual generator block with upsampling"""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        spectral_normalized_eps=0.0001,
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = paddle.nn.BatchNorm2D(num_features=input_channels)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=input_channels)
        self.relu = paddle.nn.ReLU()
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=1
            ),
            eps=spectral_normalized_eps,
        )
        self.upsample = paddle.nn.Upsample(scale_factor=2, mode="nearest")
        self.first_conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        sc = self.upsample(x)
        sc = self.conv_1x1(sc)
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.first_conv_3x3(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        x = x2 + sc
        return x


class DBlock(paddle.nn.Layer):
    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        first_relu: bool = True,
        keep_same_output: bool = False,
    ):
        """
        D and 3D Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Convolution type, see satflow/models/utils.py for options
            first_relu: Whether to have an ReLU before the first 3x3 convolution
            keep_same_output: Whether the output should have the same spatial dimensions as input, if False, downscales by 2
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        if conv_type == "3d":
            self.pooling = paddle.nn.AvgPool3D(kernel_size=2, stride=2, exclusive=False)
        else:
            self.pooling = paddle.nn.AvgPool2D(kernel_size=2, stride=2, exclusive=False)
        self.conv_1x1 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=1
            )
        )
        self.first_conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.last_conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        self.relu = paddle.nn.ReLU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x
        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)
        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x
        return x


class LBlock(paddle.nn.Layer):
    """Residual block for the Latent Stack."""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        kernel_size: int = 3,
        conv_type: str = "standard",
    ):
        """
        L-Block for increasing the number of channels in the input
         from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Which type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels - input_channels,
            kernel_size=1,
        )
        self.first_conv_3x3 = conv2d(
            input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        self.relu = paddle.nn.ReLU()
        self.last_conv_3x3 = conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )

    def forward(self, x) -> paddle.Tensor:
        if self.input_channels < self.output_channels:
            sc = self.conv_1x1(x)
            sc = paddle.concat(x=[x, sc], axis=1)
        else:
            sc = x
        x2 = self.relu(x)
        x2 = self.first_conv_3x3(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        return x2 + sc


class ContextConditioningStack(paddle.nn.Layer):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 768,
        num_context_steps: int = 4,
        conv_type: str = "standard",
        **kwargs,
    ):
        """
        Conditioning Stack using the context images from Skillful Nowcasting, , see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            output_channels: Number of output channels for the lowest block
            conv_type: Type of 2D convolution to use, see satflow/models/utils.py for options
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        output_channels = self.config["output_channels"]
        num_context_steps = self.config["num_context_steps"]
        conv_type = self.config["conv_type"]
        conv2d = get_conv_layer(conv_type)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=output_channels // 4 * input_channels // num_context_steps,
            conv_type=conv_type,
        )
        self.d2 = DBlock(
            input_channels=output_channels // 4 * input_channels // num_context_steps,
            output_channels=output_channels // 2 * input_channels // num_context_steps,
            conv_type=conv_type,
        )
        self.d3 = DBlock(
            input_channels=output_channels // 2 * input_channels // num_context_steps,
            output_channels=output_channels * input_channels // num_context_steps,
            conv_type=conv_type,
        )
        self.d4 = DBlock(
            input_channels=output_channels * input_channels // num_context_steps,
            output_channels=output_channels * 2 * input_channels // num_context_steps,
            conv_type=conv_type,
        )
        self.conv1 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=output_channels // 4 * input_channels,
                out_channels=output_channels // 8 * input_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.conv2 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=output_channels // 2 * input_channels,
                out_channels=output_channels // 4 * input_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.conv3 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=output_channels * input_channels,
                out_channels=output_channels // 2 * input_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.conv4 = paddle.nn.utils.spectral_norm(
            layer=conv2d(
                in_channels=output_channels * 2 * input_channels,
                out_channels=output_channels * input_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.relu = paddle.nn.ReLU()

    def forward(
        self, x: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        if len(x.shape) == 4:
            x = self.space2depth(x)
        elif len(x.shape) == 5:
            x_list = []
            for i in range(x.shape[0]):
                x_list.append(self.space2depth(x[i]))
            x = paddle.stack(x_list, axis=0)
        steps = x.shape[1]
        scale_1 = []
        scale_2 = []
        scale_3 = []
        scale_4 = []
        for i in range(steps):
            s1 = self.d1(x[:, i, :, :, :])
            s2 = self.d2(s1)
            s3 = self.d3(s2)
            s4 = self.d4(s3)
            scale_1.append(s1)
            scale_2.append(s2)
            scale_3.append(s3)
            scale_4.append(s4)
        scale_1 = paddle.stack(x=scale_1, axis=1)
        scale_2 = paddle.stack(x=scale_2, axis=1)
        scale_3 = paddle.stack(x=scale_3, axis=1)
        scale_4 = paddle.stack(x=scale_4, axis=1)
        scale_1 = self._mixing_layer(scale_1, self.conv1)
        scale_2 = self._mixing_layer(scale_2, self.conv2)
        scale_3 = self._mixing_layer(scale_3, self.conv3)
        scale_4 = self._mixing_layer(scale_4, self.conv4)
        return scale_1, scale_2, scale_3, scale_4

    def _mixing_layer(self, inputs, conv_block):
        stacked_inputs = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
        return paddle.nn.functional.relu(x=conv_block(stacked_inputs))


class LatentConditioningStack(paddle.nn.Layer):
    def __init__(
        self,
        shape: (int, int, int) = (8, 8, 8),
        output_channels: int = 768,
        use_attention: bool = True,
        **kwargs,
    ):
        """
        Latent conditioning stack from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            shape: Shape of the latent space, Should be (H/32,W/32,x) of the final image shape
            output_channels: Number of output channels for the conditioning stack
            use_attention: Whether to have a self-attention block or not
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        shape = self.config["shape"]
        output_channels = self.config["output_channels"]
        use_attention = self.config["use_attention"]
        self.shape = shape
        self.use_attention = use_attention
        self.distribution = paddle.distribution.Normal(
            loc=paddle.to_tensor(data=[0.0], dtype="float32"),
            scale=paddle.to_tensor(data=[2.0], dtype="float32"),
        )
        self.conv_3x3 = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=shape[0],
                out_channels=shape[0],
                kernel_size=(3, 3),
                padding=1,
            )
        )
        self.l_block1 = LBlock(
            input_channels=shape[0], output_channels=output_channels // 32
        )
        self.l_block2 = LBlock(
            input_channels=output_channels // 32, output_channels=output_channels // 16
        )
        self.l_block3 = LBlock(
            input_channels=output_channels // 16, output_channels=output_channels // 4
        )
        if self.use_attention:
            self.att_block = AttentionLayer(
                input_channels=output_channels // 4,
                output_channels=output_channels // 4,
            )
        self.l_block4 = LBlock(
            input_channels=output_channels // 4, output_channels=output_channels
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """

        Args:
            x: tensor on the correct device, to move over the latent distribution

        Returns:

        """
        z = self.distribution.sample(self.shape)
        # print('z',z.mean(),z.std())
        # import numpy as np
        # data_np = np.load('data_np.npy')
        # z = paddle.to_tensor(data_np)
        z = paddle.transpose(x=z, perm=(3, 0, 1, 2)).astype(dtype=x.dtype)
        z = self.conv_3x3(z)
        z = self.l_block1(z)
        z = self.l_block2(z)
        z = self.l_block3(z)
        z = self.att_block(z)
        z = self.l_block4(z)
        return z


def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""
    k = einops.rearrange(k, "h w c -> (h w) c")
    v = einops.rearrange(v, "h w c -> (h w) c")
    beta = paddle.nn.functional.softmax(x=paddle.einsum("hwc, Lc->hwL", q, k), axis=-1)
    out = paddle.einsum("hwL, Lc->hwc", beta, v)
    return out


class AttentionLayer(paddle.nn.Layer):
    """Attention Module"""

    def __init__(
        self, input_channels: int, output_channels: int, ratio_kq=8, ratio_v=8
    ):
        super(AttentionLayer, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.query = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=self.output_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias_attr=False,
        )
        self.key = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=self.output_channels // self.ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias_attr=False,
        )
        self.value = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=self.output_channels // self.ratio_v,
            kernel_size=(1, 1),
            padding="valid",
            bias_attr=False,
        )
        self.last_conv = paddle.nn.Conv2D(
            in_channels=self.output_channels // 8,
            out_channels=self.output_channels,
            kernel_size=(1, 1),
            padding="valid",
            bias_attr=False,
        )
        out_0 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1]).shape,
            dtype=paddle.zeros(shape=[1]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1])),
        )
        out_0.stop_gradient = not True
        self.gamma = out_0

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        out = []
        for b in range(x.shape[0]):
            out.append(attention_einsum(query[b], key[b], value[b]))
        out = paddle.stack(x=out, axis=0)
        out = self.gamma * self.last_conv(out)
        return out + x


class AddCoords(paddle.nn.Layer):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.shape
        xx_channel = paddle.arange(end=x_dim).repeat(1, y_dim, 1)
        x = paddle.arange(end=y_dim).repeat(1, x_dim, 1)
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        yy_channel = x.transpose(perm=perm_0)
        xx_channel = xx_channel.astype(dtype="float32") / (x_dim - 1)
        yy_channel = yy_channel.astype(dtype="float32") / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        x = xx_channel.repeat(batch_size, 1, 1, 1)
        perm_1 = list(range(x.ndim))
        perm_1[2] = 3
        perm_1[3] = 2
        xx_channel = x.transpose(perm=perm_1)
        x = yy_channel.repeat(batch_size, 1, 1, 1)
        perm_2 = list(range(x.ndim))
        perm_2[2] = 3
        perm_2[3] = 2
        yy_channel = x.transpose(perm=perm_2)
        ret = paddle.concat(
            x=[
                input_tensor,
                xx_channel.astype(dtype=input_tensor.dtype),
                yy_channel.astype(dtype=input_tensor.dtype),
            ],
            axis=1,
        )
        if self.with_r:
            rr = paddle.sqrt(
                x=paddle.pow(x=xx_channel.astype(dtype=input_tensor.dtype) - 0.5, y=2)
                + paddle.pow(x=yy_channel.astype(dtype=input_tensor.dtype) - 0.5, y=2)
            )
            ret = paddle.concat(x=[ret, rr], axis=1)
        return ret


class CoordConv(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = paddle.nn.Conv2D(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class ConvGRUCell(paddle.nn.Layer):
    """A ConvGRU implementation."""

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size=3, sn_eps=0.0001
    ):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps
        self.read_gate_conv = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.update_gate_conv = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.output_conv = paddle.nn.utils.spectral_norm(
            layer=paddle.nn.Conv2D(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )

    def forward(self, x, prev_state):
        """
        ConvGRU forward, returning the current+new state

        Args:
            x: Input tensor
            prev_state: Previous state

        Returns:
            New tensor plus the new state
        """
        xh = paddle.concat(x=[x, prev_state], axis=1)
        read_gate = paddle.nn.functional.sigmoid(x=self.read_gate_conv(xh))
        update_gate = paddle.nn.functional.sigmoid(x=self.update_gate_conv(xh))
        gated_input = paddle.concat(x=[x, read_gate * prev_state], axis=1)
        c = paddle.nn.functional.relu(x=self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out
        return out, new_state


class ConvGRU(paddle.nn.Layer):
    """ConvGRU Cell wrapper to replace tf.static_rnn in TF implementation"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        sn_eps=0.0001,
    ):
        super().__init__()
        self.cell = ConvGRUCell(input_channels, output_channels, kernel_size, sn_eps)

    def forward(self, x: paddle.Tensor, hidden_state=None) -> paddle.Tensor:
        outputs = []
        for step in range(len(x)):
            output, hidden_state = self.cell(x[step], hidden_state)
            outputs.append(output)
        outputs = paddle.stack(x=outputs, axis=0)
        return outputs


def get_conv_layer(conv_type: str = "standard") -> paddle.nn.Layer:
    if conv_type == "standard":
        conv_layer = paddle.nn.Conv2D
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "3d":
        conv_layer = paddle.nn.Conv3D
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = paddle.nn.functional.relu(x=1.0 - score_real)
    loss = paddle.mean(x=l1)
    l2 = paddle.nn.functional.relu(x=1.0 + score_generated)
    loss += paddle.mean(x=l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = -paddle.mean(x=score_generated)
    return loss


def grid_cell_regularizer(generated_samples, batch_targets):
    """Grid cell regularizer.

    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = paddle.mean(x=generated_samples, axis=0)
    weights = paddle.clip(x=batch_targets, min=0.0, max=24.0)
    loss = paddle.mean(x=paddle.abs(x=gen_mean - batch_targets) * weights)
    return loss