import os
import sys
import paddle
import matplotlib.pyplot as plt
import numpy as np

import paddle.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wgan_gp import WGAN_GP

class ToyGenerator(nn.Layer):
    """
    Generator network for toy datasets.
    """

    def __init__(self, noise_dim=2, output_dim=2, hidden_dim=128):
        super(ToyGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class ToyDiscriminator(nn.Layer):
    """
    Discriminator network for toy datasets.
    """

    def __init__(self, input_dim=2, hidden_dim=128):
        super(ToyDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)

class GaussianMixture(paddle.io.Dataset):
    """
    Gaussian mixture dataset for toy experiments.
    """

    def __init__(self, n_samples=10000, n_components=8, scale=2.0, std=0.2):
        super(GaussianMixture, self).__init__()

        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        centers = scale * np.column_stack((np.cos(angles), np.sin(angles)))

        samples_per_component = n_samples // n_components
        self.data = []

        for center in centers:
            samples = np.random.normal(loc=center, scale=std, size=(samples_per_component, 2))
            self.data.append(samples)

        self.data = np.vstack(self.data).astype(np.float32)
        np.random.shuffle(self.data)

        self.data = paddle.to_tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def visualize_samples(real_samples, fake_samples, save_path=None):
    """
    Visualize real and generated samples.

    Args:
        real_samples: Real data samples
        fake_samples: Generated data samples
        save_path: Path to save the visualization (default: None)
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5)
    plt.title('Real Samples')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.subplot(1, 2, 2)
    plt.scatter(fake_samples[:, 0], fake_samples[:, 1], alpha=0.5)
    plt.title('Generated Samples')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def main():
    """
    Main function to train WGAN-GP on toy dataset.
    """
    output_dir = "output/toy"
    os.makedirs(output_dir, exist_ok=True)

    dataset = GaussianMixture(n_samples=10000, n_components=8)

    generator = ToyGenerator(noise_dim=2, output_dim=2, hidden_dim=128)
    discriminator = ToyDiscriminator(input_dim=2, hidden_dim=128)

    wgan_gp = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        lambda_gp=10.0,
        critic_iters=5,
    )

    data_loader = paddle.io.DataLoader(
        dataset, batch_size=64, shuffle=True,
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

    iterations = 10000
    save_interval = 1000
    data_loader_iter = iter(data_loader)

    for iteration in range(iterations):
        try:
            real_data = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            real_data = next(data_loader_iter)

        step_results = wgan_gp.train_step(real_data, g_optimizer, d_optimizer)

        history['g_loss'].append(step_results['g_loss'])
        history['d_loss'].append(step_results['d_loss'])

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: g_loss = {step_results['g_loss']:.4f}, d_loss = {step_results['d_loss']:.4f}")

        if iteration % save_interval == 0 or iteration == iterations - 1:
            with paddle.no_grad():
                fake_samples = wgan_gp.generate(1000, noise_dim=2)

            visualize_samples(
                dataset.data.numpy(),
                fake_samples.numpy(),
                save_path=f"{output_dir}/samples_{iteration}.png",
            )

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
