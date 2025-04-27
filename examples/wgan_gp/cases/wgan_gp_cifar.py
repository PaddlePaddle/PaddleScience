import os
import sys
import paddle
import matplotlib.pyplot as plt
import paddle.nn as nn
import paddle.vision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wgan_gp import WGAN_GP

class CIFAR10Generator(nn.Layer):
    """
    Generator network for CIFAR-10 dataset.
    """

    def __init__(self, noise_dim=100, output_channels=3):
        super(CIFAR10Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 512 * 4 * 4),
            nn.BatchNorm1D(512 * 4 * 4),
            nn.ReLU(),
            lambda x: x.reshape([-1, 512, 4, 4]),
            nn.Conv2DTranspose(512, 256, 4, 2, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2DTranspose(256, 128, 4, 2, 1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2DTranspose(128, output_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

class CIFAR10Discriminator(nn.Layer):
    """
    Discriminator network for CIFAR-10 dataset.
    """

    def __init__(self, input_channels=3):
        super(CIFAR10Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2D(input_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2D(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2D(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.model(x)

def main():
    """
    Main function to train WGAN-GP on CIFAR-10 dataset.
    """
    output_dir = "output/cifar10"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = paddle.vision.datasets.Cifar10(
        mode='train', transform=transform, download=True,
    )

    generator = CIFAR10Generator(noise_dim=100, output_channels=3)
    discriminator = CIFAR10Discriminator(input_channels=3)

    wgan_gp = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        lambda_gp=10.0,
        critic_iters=5,
    )

    data_loader = paddle.io.DataLoader(
        train_dataset, batch_size=64, shuffle=True,
    )

    g_optimizer = paddle.optimizer.Adam(
        parameters=generator.parameters(),
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9,
    )

    d_optimizer = paddle.optimizer.Adam(
        parameters=discriminator.parameters(),
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9,
    )

    history = {
        'g_loss': [],
        'd_loss': [],
    }

    iterations = 50000
    save_interval = 5000
    data_loader_iter = iter(data_loader)

    for iteration in range(iterations):
        try:
            real_data = next(data_loader_iter)
            if isinstance(real_data, (list, tuple)):
                real_data = real_data[0]  # Extract images from (images, labels) tuple
        except StopIteration:
            data_loader_iter = iter(data_loader)
            real_data = next(data_loader_iter)
            if isinstance(real_data, (list, tuple)):
                real_data = real_data[0]  # Extract images from (images, labels) tuple

        step_results = wgan_gp.train_step(real_data, g_optimizer, d_optimizer)

        history['g_loss'].append(step_results['g_loss'])
        history['d_loss'].append(step_results['d_loss'])

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: g_loss = {step_results['g_loss']:.4f}, d_loss = {step_results['d_loss']:.4f}")

        if iteration % save_interval == 0 or iteration == iterations - 1:
            with paddle.no_grad():
                samples = wgan_gp.generate(16)

            from utils.visualization import save_image_grid
            save_image_grid(samples, f"{output_dir}/samples_{iteration}.png")

            paddle.save(generator.state_dict(), f"{output_dir}/generator_{iteration}.pdparams")
            paddle.save(discriminator.state_dict(), f"{output_dir}/discriminator_{iteration}.pdparams")

    plt.figure(figsize=(10, 5))
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_curves.png")
    plt.close()

if __name__ == "__main__":
    main()
