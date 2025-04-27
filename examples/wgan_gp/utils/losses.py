import paddle

def generator_loss(fake_output):
    """
    WGAN-GP generator loss function.

    Args:
        fake_output: Discriminator output for fake samples

    Returns:
        Generator loss value
    """
    return -paddle.mean(fake_output)


def discriminator_loss(real_output, fake_output, gradient_penalty, lambda_gp=10.0):
    """
    WGAN-GP discriminator loss function with gradient penalty.

    Args:
        real_output: Discriminator output for real samples
        fake_output: Discriminator output for fake samples
        gradient_penalty: Gradient penalty value
        lambda_gp: Gradient penalty coefficient (default: 10.0)

    Returns:
        Discriminator loss value
    """
    return paddle.mean(fake_output) - paddle.mean(real_output) + lambda_gp * gradient_penalty
