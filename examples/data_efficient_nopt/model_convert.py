import paddle
import numpy as np

def save_checkpoint(checkpoint_path, model):
    """Save model and optimizer to checkpoint"""

    paddle.save(
        {"model_state": model},
        checkpoint_path,
    )


def torch2paddle():
    import numpy as np
    paddle_state_dict = np.load("./checkpoint/npy_pretrain_b01_m0.tar.npy")
    save_checkpoint("./checkpoint/paddle_pretrain_b01_m0.tar", paddle_state_dict)


if __name__ == "__main__":
    torch2paddle()
