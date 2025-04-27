import paddle
import paddle.nn as nn
from .base_gan import BaseGAN

class WGAN_GP(BaseGAN):
    """
    Wasserstein GAN with Gradient Penalty implementation.
    
    This class implements the Wasserstein GAN with Gradient Penalty as described
    in the paper "Improved Training of Wasserstein GANs" by Gulrajani et al.
    """
    
    def __init__(self, generator, discriminator, lambda_gp=10.0, critic_iters=5):
        """
        Initialize the WGAN-GP with generator and discriminator networks.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            lambda_gp: Gradient penalty coefficient (default: 10.0)
            critic_iters: Number of discriminator updates per generator update (default: 5)
        """
        super(WGAN_GP, self).__init__(generator, discriminator)
        self.lambda_gp = lambda_gp
        self.critic_iters = critic_iters
        
    def generator_loss(self, fake_output):
        """
        Calculate the generator loss.
        
        Args:
            fake_output: Discriminator output for fake samples
            
        Returns:
            Generator loss value
        """
        return -paddle.mean(fake_output)
    
    def discriminator_loss(self, real_output, fake_output, gradient_penalty):
        """
        Calculate the discriminator loss with gradient penalty.
        
        Args:
            real_output: Discriminator output for real samples
            fake_output: Discriminator output for fake samples
            gradient_penalty: Gradient penalty value
            
        Returns:
            Discriminator loss value
        """
        return paddle.mean(fake_output) - paddle.mean(real_output) + self.lambda_gp * gradient_penalty
    
    def gradient_penalty(self, real_samples, fake_samples):
        """
        Calculate the gradient penalty.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated data samples
            
        Returns:
            Gradient penalty value
        """
        batch_size = real_samples.shape[0]
        
        alpha = paddle.rand(shape=[batch_size, 1, 1, 1])
        
        interpolates = real_samples + alpha * (fake_samples - real_samples)
        interpolates.stop_gradient = False
        
        disc_interpolates = self.discriminator(interpolates)
        
        gradients = paddle.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=paddle.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients_norm = paddle.sqrt(paddle.sum(paddle.square(gradients), axis=[1, 2, 3]))
        
        gradient_penalty = paddle.mean(paddle.square(gradients_norm - 1.0))
        
        return gradient_penalty
    
    def train_step(self, real_data, g_optimizer, d_optimizer):
        """
        Perform a single training step.
        
        Args:
            real_data: Batch of real data
            g_optimizer: Generator optimizer
            d_optimizer: Discriminator optimizer
            
        Returns:
            Dictionary of loss values and metrics
        """
        batch_size = real_data.shape[0]
        noise_dim = 100  # Default noise dimension
        
        d_loss_sum = 0
        for _ in range(self.critic_iters):
            noise = paddle.randn([batch_size, noise_dim])
            fake_data = self.generator(noise)
            
            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data)
            
            gp = self.gradient_penalty(real_data, fake_data)
            
            d_loss = self.discriminator_loss(real_output, fake_output, gp)
            d_loss_sum += d_loss.item()
            
            d_optimizer.clear_grad()
            d_loss.backward()
            d_optimizer.step()
        
        d_loss_avg = d_loss_sum / self.critic_iters
        
        noise = paddle.randn([batch_size, noise_dim])
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        
        g_loss = self.generator_loss(fake_output)
        
        g_optimizer.clear_grad()
        g_loss.backward()
        g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss_avg
        }
