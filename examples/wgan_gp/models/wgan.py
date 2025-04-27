import paddle
from .base_gan import BaseGAN

class WGAN(BaseGAN):
    """
    Wasserstein GAN implementation.

    This class implements the Wasserstein GAN as described in the paper
    "Wasserstein GAN" by Arjovsky et al.
    """

    def __init__(self, generator, discriminator, clip_value=0.01):
        """
        Initialize the WGAN with generator and discriminator networks.

        Args:
            generator: Generator network
            discriminator: Discriminator network
            clip_value: Value for weight clipping (default: 0.01)
        """
        super(WGAN, self).__init__(generator, discriminator)
        self.clip_value = clip_value

    def generator_loss(self, fake_output):
        """
        Calculate the generator loss.

        Args:
            fake_output: Discriminator output for fake samples

        Returns:
            Generator loss value
        """
        return -paddle.mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """
        Calculate the discriminator loss.

        Args:
            real_output: Discriminator output for real samples
            fake_output: Discriminator output for fake samples

        Returns:
            Discriminator loss value
        """
        return paddle.mean(fake_output) - paddle.mean(real_output)

    def _clip_weights(self):
        """
        Clip discriminator weights to enforce Lipschitz constraint.
        """
        for param in self.discriminator.parameters():
            param.set_value(
                paddle.clip(param, -self.clip_value, self.clip_value)
            )

    def train_step(self, real_data, g_optimizer, d_optimizer, critic_iters=5):
        """
        Perform a single training step.

        Args:
            real_data: Batch of real data
            g_optimizer: Generator optimizer
            d_optimizer: Discriminator optimizer
            critic_iters: Number of discriminator updates per generator update (default: 5)

        Returns:
            Dictionary of loss values and metrics
        """
        batch_size = real_data.shape[0]
        noise_dim = 100  # Default noise dimension

        d_loss_sum = 0
        for _ in range(critic_iters):
            noise = paddle.randn([batch_size, noise_dim])
            fake_data = self.generator(noise)

            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data)

            d_loss = self.discriminator_loss(real_output, fake_output)
            d_loss_sum += d_loss.item()

            d_optimizer.clear_grad()
            d_loss.backward()
            d_optimizer.step()

            self._clip_weights()

        d_loss_avg = d_loss_sum / critic_iters

        noise = paddle.randn([batch_size, noise_dim])
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)

        g_loss = self.generator_loss(fake_output)

        g_optimizer.clear_grad()
        g_loss.backward()
        g_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss_avg,
        }
