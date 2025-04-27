import abc

import paddle
import paddle.nn as nn

class BaseGAN(abc.ABC):
    """
    Base class for GAN implementations.
    
    This abstract class defines the common interface for all GAN variants.
    """
    
    def __init__(self, generator, discriminator):
        """
        Initialize the GAN with generator and discriminator networks.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
        """
        self.generator = generator
        self.discriminator = discriminator
        
    @abc.abstractmethod
    def generator_loss(self, fake_output):
        """
        Calculate the generator loss.
        
        Args:
            fake_output: Discriminator output for fake samples
            
        Returns:
            Generator loss value
        """
        pass
    
    @abc.abstractmethod
    def discriminator_loss(self, real_output, fake_output):
        """
        Calculate the discriminator loss.
        
        Args:
            real_output: Discriminator output for real samples
            fake_output: Discriminator output for fake samples
            
        Returns:
            Discriminator loss value
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    def generate(self, num_samples, noise_dim=100):
        """
        Generate samples using the generator.
        
        Args:
            num_samples: Number of samples to generate
            noise_dim: Dimension of the noise vector (default: 100)
            
        Returns:
            Generated samples
        """
        noise = paddle.randn([num_samples, noise_dim])
        return self.generator(noise)
    
    def train(self, train_data, batch_size=64, iterations=10000, 
              g_learning_rate=1e-4, d_learning_rate=1e-4, 
              save_interval=1000, save_path=None):
        """
        Train the GAN model.
        
        Args:
            train_data: Training dataset
            batch_size: Batch size for training (default: 64)
            iterations: Number of training iterations (default: 10000)
            g_learning_rate: Generator learning rate (default: 1e-4)
            d_learning_rate: Discriminator learning rate (default: 1e-4)
            save_interval: Interval for saving samples and model (default: 1000)
            save_path: Path to save samples and model (default: None)
            
        Returns:
            Dictionary of training history
        """
        data_loader = paddle.io.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        
        g_optimizer = paddle.optimizer.Adam(
            parameters=self.generator.parameters(),
            learning_rate=g_learning_rate,
            beta1=0.5,
            beta2=0.9
        )
        
        d_optimizer = paddle.optimizer.Adam(
            parameters=self.discriminator.parameters(),
            learning_rate=d_learning_rate,
            beta1=0.5,
            beta2=0.9
        )
        
        history = {
            'g_loss': [],
            'd_loss': []
        }
        
        data_loader_iter = iter(data_loader)
        
        for iteration in range(iterations):
            try:
                real_data = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(data_loader)
                real_data = next(data_loader_iter)
            
            step_results = self.train_step(real_data, g_optimizer, d_optimizer)
            
            history['g_loss'].append(step_results['g_loss'])
            history['d_loss'].append(step_results['d_loss'])
            
            if save_path is not None and iteration % save_interval == 0:
                samples = self.generate(16)
                
                from utils.visualization import save_image_grid
                save_image_grid(samples, f"{save_path}/samples_{iteration}.png")
                
                paddle.save(self.generator.state_dict(), f"{save_path}/generator_{iteration}.pdparams")
                paddle.save(self.discriminator.state_dict(), f"{save_path}/discriminator_{iteration}.pdparams")
        
        return history
