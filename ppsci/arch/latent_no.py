import math

import numpy as np
import paddle

from ppsci.arch import base


class LatentNO(base.Arch):
    def Attention_Vanilla(q, k, v):
        score = paddle.nn.functional.softmax(
            paddle.matmul(q, paddle.transpose(k, perm=[0, 1, 3, 2]))
            / math.sqrt(k.shape[-1]),
            axis=-1,
        )
        r = paddle.matmul(score, v)
        return r

    class LatentMLP(paddle.nn.Layer):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
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

        def forward(self, x):
            r = self.act(self.input(x))
            for i in range(0, self.n_layer):
                r = r + self.act(self.hidden[i](r))
            r = self.output(r)
            return r

    class SelfAttention(paddle.nn.Layer):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = paddle.nn.Linear(self.n_dim, self.n_dim)

        def forward(self, x):
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
        def __init__(self, n_mode, n_dim, n_head):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head

            self.self_attn = LatentNO.SelfAttention(
                self.n_mode, self.n_dim, self.n_head, LatentNO.Attention_Vanilla
            )

            self.ln1 = paddle.nn.LayerNorm(self.n_dim)
            self.ln2 = paddle.nn.LayerNorm(self.n_dim)

            self.mlp = paddle.nn.Sequential(
                paddle.nn.Linear(self.n_dim, self.n_dim * 2),
                paddle.nn.GELU(),
                paddle.nn.Linear(self.n_dim * 2, self.n_dim),
            )

        def forward(self, y):
            y1 = self.ln1(y)
            y = y + self.self_attn(y1)
            y2 = self.ln2(y)
            y = y + self.mlp(y2)
            return y

    def __init__(
        self, n_block, n_mode, n_dim, n_head, n_layer, trunk_dim, branch_dim, out_dim
    ):
        super().__init__()
        self.input_keys = ["x_y1"]  # 和 Dataset 对齐
        self.output_keys = ["y2"]
        self.trunk_dim = trunk_dim
        self.trunk_mlp = LatentNO.LatentMLP(trunk_dim, n_dim, n_dim, n_layer)
        self.branch_mlp = LatentNO.LatentMLP(branch_dim, n_dim, n_dim, n_layer)
        self.mode_mlp = LatentNO.LatentMLP(n_dim, n_dim, n_mode, n_layer)
        self.out_mlp = LatentNO.LatentMLP(n_dim, n_dim, out_dim, n_layer)

        self.attn_blocks = paddle.nn.Sequential(
            *[LatentNO.AttentionBlock(n_mode, n_dim, n_head) for _ in range(n_block)]
        )

        # Kaiming_Uniform
        for module in self.sublayers():
            if isinstance(module, paddle.nn.Linear):
                bound = 1 / math.sqrt(module.weight.shape[0])
                module.weight.set_value(
                    paddle.to_tensor(
                        np.random.uniform(-bound, bound, module.weight.shape).astype(
                            "float32"
                        )
                    )
                )
                module.bias.set_value(
                    paddle.to_tensor(
                        np.random.uniform(-bound, bound, module.bias.shape).astype(
                            "float32"
                        )
                    )
                )
            elif isinstance(module, paddle.nn.LayerNorm):
                module.weight.set_value(paddle.ones_like(module.weight))
                module.bias.set_value(paddle.zeros_like(module.bias))

    def forward(self, inputs):

        x = inputs["x"]  # (B, N, trunk_dim)
        y = inputs["y1"]  # (B, N, branch_dim)

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

        return {"y2": r}


class LatentNO_time(base.Arch):
    def Attention_Vanilla(q, k, v):
        score = paddle.nn.functional.softmax(
            paddle.matmul(q, paddle.transpose(k, perm=[0, 1, 3, 2]))
            / math.sqrt(k.shape[-1]),
            axis=-1,
        )
        r = paddle.matmul(score, v)
        return r

    class LatentMLP(paddle.nn.Layer):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
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

        def forward(self, x):
            r = self.act(self.input(x))
            for i in range(0, self.n_layer):
                r = r + self.act(self.hidden[i](r))
            r = self.output(r)
            return r

    class SelfAttention(paddle.nn.Layer):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = paddle.nn.Linear(self.n_dim, self.n_dim)

        def forward(self, x):
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
        def __init__(self, n_mode, n_dim, n_head):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head

            self.self_attn = LatentNO.SelfAttention(
                self.n_mode, self.n_dim, self.n_head, LatentNO.Attention_Vanilla
            )

            self.ln1 = paddle.nn.LayerNorm(self.n_dim)
            self.ln2 = paddle.nn.LayerNorm(self.n_dim)

            self.mlp = paddle.nn.Sequential(
                paddle.nn.Linear(self.n_dim, self.n_dim * 2),
                paddle.nn.GELU(),
                paddle.nn.Linear(self.n_dim * 2, self.n_dim),
            )

        def forward(self, y):
            y1 = self.ln1(y)
            y = y + self.self_attn(y1)
            y2 = self.ln2(y)
            y = y + self.mlp(y2)
            return y

    def __init__(
        self,
        n_block,
        n_mode,
        n_dim,
        n_head,
        n_layer,
        trunk_dim,
        branch_dim,
        out_dim,
        T=None,
        step=None,
        time_unroll=False,
    ):

        super().__init__()

        # Use separate input keys (x, y1) to align with Dataset convention.
        self.input_keys = ["x", "y1", "y2"]
        self.output_keys = ["y2"]

        # store architectural dims
        self.trunk_dim = trunk_dim
        self.branch_dim = branch_dim
        self.out_dim = out_dim

        # Single-step modules (same as original)
        self.trunk_mlp = LatentNO.LatentMLP(trunk_dim, n_dim, n_dim, n_layer)
        self.branch_mlp = LatentNO.LatentMLP(branch_dim, n_dim, n_dim, n_layer)
        self.out_mlp = LatentNO.LatentMLP(n_dim, n_dim, out_dim, n_layer)
        self.mode_mlp = LatentNO.LatentMLP(n_dim, n_dim, n_mode, n_layer)

        # time-related attributes (can be set externally before training)
        self.T = T
        self.step = step
        # --- MODIFIED ---: trunk_split indicates how to split y1 when updating during autoregression.
        # By default set to trunk_dim (but you can override before training if needed)
        self.trunk_split = trunk_dim

        self.attn_blocks = paddle.nn.Sequential(
            *[LatentNO.AttentionBlock(n_mode, n_dim, n_head) for _ in range(n_block)]
        )

        # Kaiming_Uniform
        for module in self.sublayers():
            if isinstance(module, paddle.nn.Linear):
                bound = 1 / math.sqrt(module.weight.shape[0])
                module.weight.set_value(
                    paddle.to_tensor(
                        np.random.uniform(-bound, bound, module.weight.shape).astype(
                            "float32"
                        )
                    )
                )
                module.bias.set_value(
                    paddle.to_tensor(
                        np.random.uniform(-bound, bound, module.bias.shape).astype(
                            "float32"
                        )
                    )
                )
            elif isinstance(module, paddle.nn.LayerNorm):
                module.weight.set_value(paddle.ones_like(module.weight))
                module.bias.set_value(paddle.zeros_like(module.bias))

        self.time_unroll = bool(time_unroll)
        # teacher forcing: when True *and* model.training==True, forward will use GT from inputs["y2"] as next input.
        self.use_teacher_forcing = True

    # --- MODIFIED ---: extract single-step prediction for reuse
    def _single_step_predict(self, x, y):
        """
        Single-step prediction pipeline (reuse of original forward logic).
        x: (B, N, trunk_dim)
        y: (B, N, branch_dim)
        returns r: (B, N, out_dim)
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

    def forward(self, inputs):
        """
        inputs: dict with keys 'x', 'y1' and optionally 'y2' (GT sequence) when using time_unroll.
        Returns:
          if time_unroll == False:
              {"y2": r} with r shape (B, N, out_dim)
          if time_unroll == True:
              {"y2": pred_full, "y2_steps": pred_steps_stack}
                pred_full shape: (B, N, T_total) (concatenated along time dim, same layout as y2 GT)
                pred_steps_stack shape: (B, N, step, num_steps) (stacked per-step predictions)
        """

        x = inputs["x"]
        y1 = inputs["y1"]
        # optional GT sequence (only present if dataset put it into inputs)
        y2_gt = inputs.get("y2", None)

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

        return {"y2": pred_full, "y2_steps": pred_steps_stack}
