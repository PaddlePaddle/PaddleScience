import numpy as np
import matplotlib.pyplot as plt
import paddle

def save_image_grid(images, path, nrow=8, padding=2, normalize=True):
    """
    Save a grid of images to a file.
    
    Args:
        images: Tensor of images to display
        path: Path to save the image grid
        nrow: Number of images per row (default: 8)
        padding: Padding between images (default: 2)
        normalize: Whether to normalize images to [0, 1] (default: True)
    """
    if isinstance(images, paddle.Tensor):
        images = images.numpy()
    
    if normalize:
        images = (images - images.min()) / (images.max() - images.min() + 1e-8)
    
    nmaps = images.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(images.shape[1] + padding), int(images.shape[2] + padding)
    
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, 3), dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            image = images[k]
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            image = (image * 255).astype(np.uint8)
            grid[y * height + padding:(y + 1) * height, 
                 x * width + padding:(x + 1) * width] = image
            k += 1
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_loss_curves(g_losses, d_losses, path):
    """
    Plot generator and discriminator loss curves.
    
    Args:
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
