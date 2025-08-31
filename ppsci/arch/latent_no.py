import math
from typing import Callable
from typing import Optional
from typing import Tuple

import paddle

from ppsci.arch import base
from ppsci.utils import initializer

AttentionFn = Callable[[paddle.Tensor, paddle.Tensor, paddle.Tensor], paddle.Tensor]


def Attention_Vanilla(
    q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor
) -> paddle.Tensor:
    """
    Args:
        q (paddle.Tensor): Query tensor of shape (B, n_head, N, D_k).
        k (paddle.Tensor): Key tensor of shape (B, n_head, N, D_k).
        v (paddle.Tensor): Value tensor of shape (B, n_head, N, D_v).

    Returns:
        paddle.Tensor: Attention output of shape (B, n_head, N, D_v).
    """
    score = paddle.nn.functional.softmax(
        paddle.matmul(q, paddle.transpose(k, perm=[0, 1, 3, 2]))
        / math.sqrt(k.shape[-1]),
        axis=-1,
    )
    r = paddle.matmul(score, v)
    return r


class LatentMLP(paddle.nn.Layer):
    """
    Multi-layer perceptron with residual connections used for trunk/branch/mode/out projections.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension.
        output_dim (int): Output feature dimension.
        n_layer (int): Number of hidden layers (residual blocks).

    Input:
        x (paddle.Tensor): shape (B, N, input_dim) or (..., input_dim)

    Returns:
        paddle.Tensor: shape (B, N, output_dim)
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_layer: int
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.act = paddle.nn.GELU()
        self.input = paddle.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = paddle.nn.LayerList(
            [
                paddle.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_layer)
            ]
        )
        self.output = paddle.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x (paddle.Tensor): Input tensor of shape (B, N, input_dim).

        Returns:
            paddle.Tensor: Output tensor of shape (B, N, output_dim).
        """
        r = self.act(self.input(x))
        for i in range(0, self.n_layer):
            r = r + self.act(self.hidden[i](r))
        r = self.output(r)
        return r


class SelfAttention(paddle.nn.Layer):
    """
    Multi-head self-attention module.

    Args:
        n_mode (int): Sequence length / number of modes (used for documentation only).
        n_dim (int): Input/output feature dimension (D).
        n_head (int): Number of attention heads.
        attn (Callable): Attention function with signature (q, k, v) -> out.

    Input:
        x (paddle.Tensor): shape (B, N, D)

    Returns:
        paddle.Tensor: shape (B, N, D)
    """

    def __init__(self, n_mode: int, n_dim: int, n_head: int, attn: AttentionFn) -> None:
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.Wq = paddle.nn.Linear(self.n_dim, self.n_dim)
        self.Wk = paddle.nn.Linear(self.n_dim, self.n_dim)
        self.Wv = paddle.nn.Linear(self.n_dim, self.n_dim)
        self.attn = attn
        self.proj = paddle.nn.Linear(self.n_dim, self.n_dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x (paddle.Tensor): Input tensor of shape (B, N, D).

        Returns:
            paddle.Tensor: Output tensor of shape (B, N, D).
        """
        B, N, D = tuple(x.shape)
        q = self.Wq(x)
        q = paddle.reshape(q, (B, N, self.n_head, D // self.n_head))
        q = paddle.transpose(q, [0, 2, 1, 3])
        k = self.Wk(x)
        k = paddle.reshape(k, (B, N, self.n_head, D // self.n_head))
        k = paddle.transpose(k, [0, 2, 1, 3])
        v = self.Wv(x)
        v = paddle.reshape(v, (B, N, self.n_head, D // self.n_head))
        v = paddle.transpose(v, [0, 2, 1, 3])
        r = self.attn(q, k, v)
        r = paddle.transpose(r, [0, 2, 1, 3])
        r = paddle.reshape(r, (B, N, D))
        r = self.proj(r)
        return r


class AttentionBlock(paddle.nn.Layer):
    """
    Transformer-style block: LayerNorm -> Self-Attention (residual) -> LayerNorm -> MLP (residual).

    Args:
        n_mode (int): Sequence length / number of modes (documentation).
        n_dim (int): Feature dimension D.
        n_head (int): Number of attention heads.

    Input:
        y (paddle.Tensor): shape (B, N, D)

    Returns:
        paddle.Tensor: shape (B, N, D)
    """

    def __init__(self, n_mode: int, n_dim: int, n_head: int) -> None:
        super().__init__()
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head

        self.self_attn = SelfAttention(
            self.n_mode, self.n_dim, self.n_head, Attention_Vanilla
        )

        self.ln1 = paddle.nn.LayerNorm(self.n_dim)
        self.ln2 = paddle.nn.LayerNorm(self.n_dim)

        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(self.n_dim, self.n_dim * 2),
            paddle.nn.GELU(),
            paddle.nn.Linear(self.n_dim * 2, self.n_dim),
        )

    def forward(self, y: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass of the Transformer-style attention block.

        Args:
            y (paddle.Tensor): Input tensor of shape (B, N, D).

        Returns:
            paddle.Tensor: Output tensor of shape (B, N, D).
        """
        y1 = self.ln1(y)
        y = y + self.self_attn(y1)
        y2 = self.ln2(y)
        y = y + self.mlp(y2)
        return y


class LatentNO(base.Arch):
    def __init__(
        self,
        n_block: int,
        n_mode: int,
        n_dim: int,
        n_head: int,
        n_layer: int,
        trunk_dim: int,
        branch_dim: int,
        out_dim: int,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
    ):
        """
        Latent Neural Operator (LatentNO).

        Args:
            n_block (int): Number of attention blocks.
            n_mode (int): Number of latent modes.
            n_dim (int): Hidden dimension size.
            n_head (int): Number of attention heads.
            n_layer (int): Number of layers in MLP.
            trunk_dim (int): Dimension of trunk input.
            branch_dim (int): Dimension of branch input.
            out_dim (int): Dimension of output.
            input_keys (Tuple[str, ...]): Name of input keys.
            output_keys (Tuple[str, ...]): Name of output keys.
        """
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.trunk_dim = trunk_dim
        self.trunk_mlp = LatentMLP(trunk_dim, n_dim, n_dim, n_layer)
        self.branch_mlp = LatentMLP(branch_dim, n_dim, n_dim, n_layer)
        self.mode_mlp = LatentMLP(n_dim, n_dim, n_mode, n_layer)
        self.out_mlp = LatentMLP(n_dim, n_dim, out_dim, n_layer)

        self.attn_blocks = paddle.nn.Sequential(
            *[AttentionBlock(n_mode, n_dim, n_head) for _ in range(n_block)]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, paddle.nn.Linear):
            initializer.linear_init_(module)
        elif isinstance(module, paddle.nn.Conv2D):
            initializer.conv_init_(module)
        elif isinstance(module, paddle.nn.LayerNorm):
            initializer.ones_(module.weight)
            initializer.zeros_(module.bias)

    def forward(self, inputs: dict[str, paddle.Tensor]) -> dict[str, paddle.Tensor]:
        """
        Forward pass of LatentNO.

        Args:
            inputs (dict[str, paddle.Tensor]):
                Dictionary with keys:
                    - "x": Trunk input tensor of shape (B, N, trunk_dim).
                    - "y1": Branch input tensor of shape (B, N, branch_dim).

        Returns:
            dict[str, paddle.Tensor]: Dictionary containing:
                - "y2": Output tensor of shape (B, N, out_dim).
        """
        x = inputs[self.input_keys[0]]  # trunk input
        y = inputs[self.input_keys[1]]  # branch input

        x = self.trunk_mlp(x)
        y = self.branch_mlp(y)

        score = self.mode_mlp(x)
        score_encode = paddle.nn.functional.softmax(score, axis=1)
        score_decode = paddle.nn.functional.softmax(score, axis=-1)

        z = paddle.matmul(paddle.transpose(score_encode, perm=[0, 2, 1]), y)
        for block in self.attn_blocks:
            z = block(z)

        r = paddle.matmul(score_decode, z)
        r = self.out_mlp(r)

        return {self.output_keys[0]: r}


class LatentNO_time(base.Arch):
    """
    Time-enabled Latent Neural Operator.

    This class implements the time-unrolled (autoregressive) variant of LatentNO.
    It supports both single-step prediction and multi-step autoregressive unrolling
    controlled by the `time_unroll` flag, `T` (total steps) and `step` (step size).
    """

    def __init__(
        self,
        n_block: int,
        n_mode: int,
        n_dim: int,
        n_head: int,
        n_layer: int,
        trunk_dim: int,
        branch_dim: int,
        out_dim: int,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        T: Optional[int] = None,
        step: Optional[int] = None,
        time_unroll: bool = False,
    ) -> None:
        """
        Initialize LatentNO_time.

        Args:
            n_block (int): Number of attention blocks.
            n_mode (int): Number of latent modes.
            n_dim (int): Hidden dimension size.
            n_head (int): Number of attention heads.
            n_layer (int): Number of layers in MLP.
            trunk_dim (int): Dimension of trunk input.
            branch_dim (int): Dimension of branch input.
            out_dim (int): Dimension of output.
            input_keys (Tuple[str, ...]): Name of input keys.
            output_keys (Tuple[str, ...]): Name of output keys.
            T (Optional[int]): Total number of time steps for unrolled mode.
            step (Optional[int]): Time step size used during unrolling.
            time_unroll (bool): Whether to enable autoregressive time unrolling.
        """
        super().__init__()

        # Use separate input keys (x, y1) to align with Dataset convention.
        self.input_keys = input_keys
        self.output_keys = output_keys

        # store architectural dims
        self.trunk_dim = trunk_dim
        self.branch_dim = branch_dim
        self.out_dim = out_dim

        # Single-step modules (same as original)
        self.trunk_mlp = LatentMLP(trunk_dim, n_dim, n_dim, n_layer)
        self.branch_mlp = LatentMLP(branch_dim, n_dim, n_dim, n_layer)
        self.out_mlp = LatentMLP(n_dim, n_dim, out_dim, n_layer)
        self.mode_mlp = LatentMLP(n_dim, n_dim, n_mode, n_layer)

        # time-related attributes (can be set externally before training)
        self.T = T
        self.step = step
        # --- MODIFIED ---: trunk_split indicates how to split y1 when updating during autoregression.
        # By default set to trunk_dim (but you can override before training if needed)
        self.trunk_split = trunk_dim

        self.attn_blocks = paddle.nn.Sequential(
            *[AttentionBlock(n_mode, n_dim, n_head) for _ in range(n_block)]
        )

        self.apply(self._init_weights)

        self.time_unroll = bool(time_unroll)
        # teacher forcing: when True *and* model.training==True, forward will use GT from inputs["y2"] as next input.
        self.use_teacher_forcing = True

    def _init_weights(self, module):
        if isinstance(module, paddle.nn.Linear):
            initializer.linear_init_(module)
        elif isinstance(module, paddle.nn.Conv2D):
            initializer.conv_init_(module)
        elif isinstance(module, paddle.nn.LayerNorm):
            initializer.ones_(module.weight)
            initializer.zeros_(module.bias)

    # Extract single-step prediction for reuse
    def _single_step_predict(self, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        """
        Compute single-step prediction (reused in forward).

        Args:
            x (paddle.Tensor): Trunk input tensor of shape (B, N, trunk_dim).
            y (paddle.Tensor): Branch input tensor of shape (B, N, branch_dim).

        Returns:
            paddle.Tensor: Output tensor of shape (B, N, out_dim).
        """
        x_enc = self.trunk_mlp(x)
        y_enc = self.branch_mlp(y)

        score = self.mode_mlp(x_enc)
        score_encode = paddle.nn.functional.softmax(score, axis=1)
        score_decode = paddle.nn.functional.softmax(score, axis=-1)

        z = paddle.matmul(paddle.transpose(score_encode, perm=[0, 2, 1]), y_enc)
        for block in self.attn_blocks:
            z = block(z)

        r = paddle.matmul(score_decode, z)
        r = self.out_mlp(r)

        return r

    def forward(self, inputs: dict[str, paddle.Tensor]) -> dict[str, paddle.Tensor]:
        """
        Forward pass of LatentNO_time.

        Args:
            inputs (dict[str, paddle.Tensor]):
                Dictionary with keys:
                    - "x": Trunk input tensor of shape (B, N, trunk_dim).
                    - "y1": Branch input tensor of shape (B, N, branch_dim).
                    - "y2" (optional): Ground-truth sequence for teacher forcing (B, N, T).

        Returns:
            dict[str, paddle.Tensor]:
                - If time_unroll == False:
                    {"y2": (B, N, out_dim)}
                - If time_unroll == True:
                    {"y2": (B, N, T_total), "y2_steps": (B, N, step, num_steps)}
        """
        x = inputs[self.input_keys[0]]
        y1 = inputs[self.input_keys[1]]
        y2_gt = inputs.get(self.input_keys[2], None)  # optional ground truth

        # simple single-step (original behaviour)
        if not getattr(self, "time_unroll", False):
            r = self._single_step_predict(x, y1)
            return {"y2": r}

        # time-unroll (autoregressive) mode
        if self.T is None or self.step is None:
            raise ValueError("time_unroll enabled but model.T or model.step is None.")
        if not hasattr(self, "trunk_split") or self.trunk_split is None:
            raise ValueError("time_unroll enabled but model.trunk_split is not set.")

        current_y = y1
        pred_steps = []

        # iterate time: mimic original `for t in range(0, T, step)`
        for t in range(0, self.T, self.step):
            # predict one step
            pred_step = self._single_step_predict(x, current_y)

            # append for final concatenation
            pred_steps.append(pred_step)

            # - training + use_teacher_forcing -> use GT slice from inputs["y2"] (teacher forcing)
            # - otherwise -> use pred_step (autoregressive)
            if (
                self.training
                and getattr(self, "use_teacher_forcing", False)
                and (y2_gt is not None)
            ):
                # use GT slice (must exist and have time alignment)
                next_input_part = y2_gt[..., t : t + self.step]
            else:
                # use prediction as next input; make sure to block gradient so predictions don't backprop through time
                pred_step.stop_gradient = True
                next_input_part = pred_step

            # update current_y: keep trunk part, drop earliest step slot(s), append the next part
            left = current_y[..., : self.trunk_split]
            right = current_y[..., self.trunk_split + self.step :]
            current_y = paddle.concat((left, right, next_input_part), axis=-1)

        # final outputs: concat along time dimension (last dim of out is per-step time dim)
        pred_full = paddle.concat(pred_steps, axis=-1)
        pred_steps_stack = paddle.stack(pred_steps, axis=-1)

        return {
            self.output_keys[0]: pred_full,
            f"{self.output_keys[0]}_steps": pred_steps_stack,
        }
