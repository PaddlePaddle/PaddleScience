import os
import sys
import paddle
import matplotlib.pyplot as plt
import numpy as np
import paddle.nn as nn
from ..models.wgan_gp import WGAN_GP

class MNISTGenerator(nn.Layer):
    """
    Generator network for MNIST dataset.
    """

    def __init__(self, noise_dim=100, output_channels=1):
        super(MNISTGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128 * 7 * 7),
            nn.BatchNorm1D(128 * 7 * 7),
            nn.ReLU(),
            lambda x: x.reshape([-1, 128, 7, 7]),
            nn.Conv2DTranspose(128, 64, 4, 2, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2DTranspose(64, output_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

class MNISTDiscriminator(nn.Layer):
    """
    Discriminator network for MNIST dataset.
    """

    def __init__(self, input_channels=1):
        super(MNISTDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2D(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2D(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.model(x)

def main():
    """
    Main function to train WGAN-GP on MNIST dataset.
    """
    output_dir = "output/mnist"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = paddle.vision.datasets.MNIST(
        mode='train', transform=transform, download=True,
    )

    generator = MNISTGenerator(noise_dim=100, output_channels=1)
    discriminator = MNISTDiscriminator(input_channels=1)

    wgan_gp = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        lambda_gp=10.0,
        critic_iters=5,
    )

    history = wgan_gp.train(
        train_dataset,
        batch_size=64,
        iterations=20000,
        g_learning_rate=1e-4,
        d_learning_rate=1e-4,
        save_interval=1000,
        save_path=output_dir,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_curves.png")
    plt.close()

    with paddle.no_grad():
        samples = wgan_gp.generate(16)

    from utils.visualization import save_image_grid
    save_image_grid(samples, f"{output_dir}/final_samples.png")

if __name__ == "__main__":
    main()
