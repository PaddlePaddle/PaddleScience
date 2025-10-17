import paddle
import paddle.nn as nn


class PositionalEncoder(nn.Layer):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = paddle.pow(
            paddle.to_tensor(T, dtype="float32"),
            2 * (paddle.arange(offset, offset + d, dtype="float32") // 2) / d,
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            # Move to same device as input
            if hasattr(batch_positions, "place"):
                self.denom = (
                    self.denom.cuda()
                    if "gpu" in str(batch_positions.place)
                    else self.denom.cpu()
                )
            self.updated_location = True

        sinusoid_table = (
            batch_positions[:, :, None].astype("float32") / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = paddle.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = paddle.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = paddle.concat(
                [sinusoid_table for _ in range(self.repeat)], axis=-1
            )

        return sinusoid_table
