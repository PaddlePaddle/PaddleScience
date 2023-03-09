# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
This code is refer from:
https://github.com/zabaras/transformer-physx/blob/main/trphysx/transformer/attention.py
https://github.com/zabaras/transformer-physx/blob/main/trphysx/transformer/phys_transformer_gpt2.py
"""

import os

import paddle
from paddle import nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from .generate_utils import GenerationMixin

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

Tensor = paddle.Tensor


class MaskedAttention(nn.Layer):
    """ Masked self-attention module """

    def __init__(
            self,
            nx,
            n_ctx,
            n_head,
            attn_drop=0.,
            proj_drop=0.,
            scale=False, ):
        super().__init__()

        n_state = nx
        assert n_state % n_head == 0

        # Create attention mask
        self.register_buffer(
            "bias",
            paddle.tril(paddle.ones(
                (n_ctx, n_ctx), dtype='int32')).reshape([1, 1, n_ctx, n_ctx]))

        self.register_buffer("masked_bias", paddle.to_tensor(-1e4))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = nn.Linear(nx, n_state)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pruned_heads = set()

    def _attn(self,
              q,
              k,
              v,
              attention_mask=None,
              head_mask=None,
              output_attentions=False):
        attn = paddle.matmul(q, k)
        if self.scale:
            attn = attn / (float(v.shape[-1])**0.5)

        nd, ns = attn.shape[-2], attn.shape[-1]
        mask = self.bias[:, :, ns - nd:ns, :ns]
        attn = paddle.where(mask > 0, attn, self.masked_bias.cast(attn.dtype))

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        if head_mask is not None:
            attn = attn * head_mask

        outputs = [paddle.matmul(attn, v)]
        if output_attentions:
            outputs.append(attn)
        return outputs

    def merge_heads(self, x):
        """Merge attention heads

        Args:
            x (Tensor): [batch, head, seq_length, head_features] Input tensor

        Returns:
            Tensor: [batch, seq_length, head * head_features] Concatenated output tensor
        """
        x = x.transpose([0, 2, 1, 3])
        new_x_shape = x.shape[:-2] + [x.shape[-2] * x.shape[-1], ]
        return x.reshape(new_x_shape)

    def split_heads(self, x, k=False):
        """Splits key, query or value tensor into separate heads.
        Dimensionality of output depends if tensor is a key.

        Args:
            x (Tensor): [batch, seq_length, nx] Input tensor
            k (bool): If input tensor is a key tensor

        Returns:
            Tensor: [batch, head, seq_length, head_features] Split features for query
            and value, [batch, head, seq_length, head_features] split feature for key
        """
        new_x_shape = x.shape[:-1] + [self.n_head, x.shape[-1] // self.n_head]
        x = x.reshape(new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.transpose(
                [0, 2, 3, 1])  # (batch, head, head_features, seq_length)
        else:
            return x.transpose(
                [0, 2, 1, 3])  # (batch, head, seq_length, head_features)

    def forward(self,
                x,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False):
        """Masked attention forward pass

        Args:
            x (Tensor): [batch, seq_length, nx] Input feature.
            layer_past (Tensor, optional): [2, batch, n_head, seq_length, nx] Precomputed self-attention vectors. Defaults to None.
            attention_mask (Tensor, optional): Optional defined attention mask. Applied before soft mask.
                 Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Applied after softmax Defaults to None.
            use_cache (bool, optional): Return calculated key values or faster generation. Defaults to False.
            output_attentions (bool, optional): Return attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, key values (if requested), attention tensor (if requested)
        """
        x = self.c_attn(x)  # x -> q, k, v
        query, key, value = x.split(x.shape[2] // self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(
            key, k=True)  # k=True for keys which transposes the last two dims
        value = self.split_heads(value)
        # Concat previous key and value tensors
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(
                [0, 1, 3, 2]), layer_past[1]  # transpose back cf below
            key = paddle.concat((past_key, key), axis=-1)
            value = paddle.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = paddle.stack(
                (key.transpose([0, 1, 3, 2]),
                 value))  # transpose to have same shapes for stacking
        else:
            present = (None, )

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask,
                                  output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.proj_drop(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Layer):
    """Transformer decoder block consisting of layer norm, masked self-attention,
    layer norm and fully connected layer.

    Args:
        n_ctx (int): contex length of block
        config (PhysConfig): Phys-transformer config object
        scale (bool, optional): Scaled self-attention calculation. Defaults to False.
    """

    def __init__(self,
                 n_ctx: int,
                 n_embd,
                 n_head,
                 attn_pdrop,
                 resid_pdrop,
                 scale=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MaskedAttention(n_embd, n_ctx, n_head, attn_pdrop,
                                    resid_pdrop, scale)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, 4 * n_embd, resid_pdrop)

    def forward(
            self,
            x,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False, ):
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features
            layer_past ([type], optional): Past self-attention calculation. Defaults to None.
            attention_mask (LongTensor, optional): Attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value. Defaults to None.
            use_cache (bool, optional): Store attention state (key values). Defaults to False.
            output_attentions (bool, optional): Return attention values. Defaults to False.

        Returns:
            List[Tensor]: List of output tensors
        """
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions, )

        a = output_attn[0]
        # Residual connection 1
        x = x + a
        # FCNN
        m = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class PhysformerGPT2(GenerationMixin,
                     nn.Layer):  # Mixins come first before base to overload
    """Transformer decoder model for modeling physics
    """

    def __init__(
            self,
            n_layer,
            n_ctx,
            n_embd,
            n_head,
            embedding_model,
            pretrained_model=None,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            initializer_range=0.05,
            viz=None, ):
        nn.Layer.__init__(self)
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.initializer_range = initializer_range
        self.viz = viz

        self.embedding_model = embedding_model

        self.output_hidden_states = False
        self.drop = nn.Dropout(embd_pdrop)
        self.h = nn.LayerList([
            Block(
                n_ctx, n_embd, n_head, attn_pdrop, resid_pdrop, scale=True)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.mlp_f = nn.Linear(n_embd, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)

        self.apply(self._init_weights)
        if pretrained_model is not None:
            state_dict = paddle.load(pretrained_model)
            self.embedding_model.set_state_dict(state_dict)
        self.embedding_model.eval()

        self.n_embd = n_embd

        self.loss_fun = nn.MSELoss()

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            normal_ = Normal(mean=0.0, std=self.initializer_range)
            normal_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)

    def forward(self,
                inputs_embeds,
                position_ids=None,
                prop_embeds=None,
                past=None,
                attention_mask=None,
                head_mask=None,
                use_cache=True,
                output_attentions=False,
                **kwargs):

        # Input embeddings
        input_shape = inputs_embeds.shape[:-1]
        batch_size = inputs_embeds.shape[0]

        if position_ids is not None:
            position_ids = position_ids.reshpae([-1, input_shape[-1]])

        if prop_embeds is not None:
            assert inputs_embeds.shape[0] == prop_embeds.shape[
                0], 'Property embeddings do not match the size of the input'
            prop_embeds = prop_embeds[:, :inputs_embeds.shape[1]]
        else:
            prop_embeds = paddle.zeros_like(inputs_embeds)

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].shape[-2]

        if position_ids is None:
            position_ids = paddle.arange(past_length,
                                         input_shape[-1] + past_length)
            position_ids = position_ids.unsqueeze(0).reshape(
                [-1, input_shape[-1]]).repeat_interleave(
                    inputs_embeds.shape[0], axis=0)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.reshape([batch_size, -1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Set mask to 0 for positions we want to attend and -10000 for ones we do not
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Function embeddings proposed in original transformer paper
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        position_embeds = paddle.zeros_like(inputs_embeds)
        i = paddle.arange(0, self.n_embd // 2).unsqueeze(0).unsqueeze(0)
        position_embeds[:, :, ::2] = paddle.sin(
            position_ids.unsqueeze(-1).cast(paddle.float32) / 10000
            **(2 * i / self.n_embd))
        i = i[:, :, self.n_embd % 2]
        position_embeds[:, :, 1::2] = paddle.cos(
            position_ids.unsqueeze(-1).cast(paddle.float32) / 10000
            **(2 * i / self.n_embd))

        # Combine input embedding, position embeding and prop embeddings
        hidden_states = inputs_embeds + position_embeds + prop_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + [hidden_states.shape[-1]]

        # Loop through transformer self-attention layers
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    hidden_states.reshape(*output_shape), )

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions, )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present, )

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.mlp_f(self.ln_f(hidden_states))

        hidden_states = hidden_states.reshape(output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if use_cache is True:
            outputs = outputs + (presents, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [
                -1,
            ] + all_attentions[0].shape[-2:]
            all_attentions = tuple(
                t.reshape(attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions, )

        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)

    def compute_loss(self, inputs, **kwargs):
        # states: b, n, d = 16, 64, 32
        with paddle.no_grad():
            embedding_data = self.embedding_model.embed(inputs, **kwargs)
        inputs_embeds = embedding_data[:, :-1]  # 16, 63, 32
        labels_embeds = embedding_data[:, 1:]  # 16, 63, 32

        outputs = self.forward(inputs_embeds=inputs_embeds, **kwargs)
        # compute loss
        hidden_states = outputs[0]
        # Flatten the tokens
        loss = self.loss_fun(hidden_states, labels_embeds)

        return dict(loss=loss)

    def evaluate(self, inputs, visu_dir=None, **kwargs):
        self.eval()
        # states: b, n, d = 16, 64, 32
        with paddle.no_grad():
            embedding_data = self.embedding_model.embed(inputs, **kwargs)
        inputs_embeds = embedding_data[:, :1]  # 16, 1, 32
        labels_embeds = embedding_data  # 16, 64, 32

        max_length = labels_embeds.shape[1]

        outputs = self.generate(
            inputs_embeds=inputs_embeds, max_length=max_length, **kwargs)
        pred_embeds = outputs[0]

        bsize = pred_embeds.shape[0]
        tsize = pred_embeds.shape[1]

        x_in = pred_embeds.reshape([-1, pred_embeds.shape[-1]])
        out = self.embedding_model.recover(x_in)
        out = out.reshape([bsize, tsize] + self.embedding_model.state_dims)

        state_error = self.loss_fun(out, inputs)

        if self.viz is not None:
            if visu_dir is not None:
                os.makedirs(visu_dir, exist_ok=True)

            for i in range(bsize):
                self.viz.plotPrediction(out[i], inputs[i], visu_dir, pid=i)

        error = self.loss_fun(pred_embeds, labels_embeds)

        return dict(
            loss=error,
            state_error=state_error,
            pred_embeds=pred_embeds,
            labels_embeds=labels_embeds)
