import math

import paddle
from paddle import nn

# MoE Gating


class GatingNet(nn.Layer):
    def __init__(self, moe_config, input_shape, in_channels):
        super().__init__()

        self.num_experts = moe_config["num_experts"]
        self.out_planes = moe_config["out_planes"]
        self.aux_loss_style = moe_config["aux_loss_style"]
        assert self.out_planes > 1 and self.out_planes <= self.num_experts
        assert len(input_shape) == 4
        self.input_shape = input_shape

        self.noise_lin = nn.Linear(
            in_features=in_channels, out_features=self.num_experts, bias_attr=False
        )
        self.noise_eps = 1e-2
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(axis=-1)

        self.importance_weight = moe_config["importance_weight"]
        self.load_weight = moe_config["load_weight"]

    def cv_squared(self, x, eps=1e-25):
        return x.var(axis=-1) / (x.mean(axis=-1) ** 2 + eps)

    def intra_cdf(self, value, loc=0.0, scale=1.0):
        return 0.5 * (1 + paddle.erf((value - loc) / scale / math.sqrt(2)))

    def importance_loss_cell(self, routing_weights):
        importance_loss = self.cv_squared(routing_weights.sum(axis=0)).mean()
        return importance_loss

    def load_loss_cell(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        B, T, H, W, E = clean_values.shape
        M = noisy_top_values.shape[-1]
        clean_values = clean_values.transpose([1, 2, 3, 0, 4])
        noisy_values = noisy_values.transpose([1, 2, 3, 0, 4])
        noise_stddev = noise_stddev.transpose([1, 2, 3, 0, 4])
        top_values_flat = noisy_top_values.transpose([1, 2, 3, 0, 4]).reshape(
            [T, H, W, B * M]
        )

        threshold_positions_if_in = paddle.arange(B) * M + self.out_planes
        threshold_if_in = paddle.take_along_axis(
            top_values_flat,
            axis=-1,
            indices=threshold_positions_if_in.unsqueeze(axis=[0, 1, 2]),
        ).unsqueeze(
            -1
        )  # T, H, W, B, 1
        is_in = noisy_values > threshold_if_in  # T, H, W, B, E
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = paddle.take_along_axis(
            top_values_flat,
            axis=-1,
            indices=threshold_positions_if_out.unsqueeze(axis=[0, 1, 2]),
        ).unsqueeze(-1)

        prob_if_in = self.intra_cdf(
            (clean_values - threshold_if_in) / noise_stddev
        )  # T, H, W, B, E
        prob_if_out = self.intra_cdf(
            (clean_values - threshold_if_out) / noise_stddev
        )  # T, H, W, B, E
        prob = paddle.where(is_in, prob_if_in, prob_if_out)  # T, H, W, B, E

        load_loss = self.cv_squared(prob.sum(axis=-2)).mean()
        return load_loss

    def importance_loss_all(self, routing_weights):
        importance_loss = self.cv_squared(routing_weights.sum(axis=0))
        return importance_loss

    def load_loss_all(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        B, E = clean_values.shape
        M = noisy_top_values.shape[-1]
        top_values_flat = noisy_top_values.flatten()  # B * M

        threshold_positions_if_in = paddle.arange(B) * M + self.out_planes  # B
        threshold_if_in = paddle.take_along_axis(
            top_values_flat, axis=-1, indices=threshold_positions_if_in
        ).unsqueeze(
            -1
        )  # B, 1
        is_in = noisy_values > threshold_if_in  # B, E
        threshold_positions_if_out = threshold_positions_if_in - 1  # B
        threshold_if_out = paddle.take_along_axis(
            top_values_flat, axis=-1, indices=threshold_positions_if_out
        ).unsqueeze(
            -1
        )  # B, 1

        prob_if_in = self.intra_cdf(
            (clean_values - threshold_if_in) / noise_stddev
        )  # B, E
        prob_if_out = self.intra_cdf(
            (clean_values - threshold_if_out) / noise_stddev
        )  # B, E
        prob = paddle.where(is_in, prob_if_in, prob_if_out)  # B, E

        load_loss = self.cv_squared(prob.sum(axis=0))
        return load_loss

    def forward(self, x, t_map=None, eps=1e-25, dense_routing=False):
        assert x.shape[1:-1] == list(self.input_shape)[:-1]
        B, T, H, W, C = x.shape
        E = self.num_experts

        raw_logits = self.gating(x, t_map)
        if self.training:
            noise = self.softplus(self.noise_lin(x)) + self.noise_eps
            noisy_logits = raw_logits + paddle.randn(shape=raw_logits.shape) * noise
            logits = noisy_logits
        else:
            logits = raw_logits

        assert logits.shape[-1] == self.num_experts
        logits = self.softmax(logits)  # [B, T, H, W, E]
        top_logits, top_indices = logits.topk(
            min(self.out_planes + 1, self.num_experts), axis=-1
        )
        top_k_logits = top_logits[:, :, :, :, : self.out_planes]
        top_k_indices = top_indices[:, :, :, :, : self.out_planes]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(axis=-1, keepdim=True) + eps
        )  # normalization

        if dense_routing:
            # zeros = paddle.zeros_like(logits)
            # zeros.stop_gradient = False
            # print(zeros.shape)
            # print(top_k_gates.shape, top_k_gates[0, 0, 0, 0])
            # routing_weights = paddle.put_along_axis(zeros, axis=-1, indices=top_k_indices, values=top_k_gates)
            # print(routing_weights.shape, routing_weights.stop_gradient)
            pass
        else:
            routing_weights = None

        if self.training:
            if self.aux_loss_style == "cell":
                # importance_loss = self.importance_loss(routing_weights)
                importance_loss = self.importance_loss_cell(logits)
                load_loss = self.load_loss_cell(
                    raw_logits, noisy_logits, noise, top_logits
                )
            elif self.aux_loss_style == "all":
                importance_loss = self.importance_loss_all(
                    logits.reshape([B * T * H * W, E])
                )
                load_loss = self.load_loss_all(
                    raw_logits.reshape([B * T * H * W, E]),
                    noisy_logits.reshape([B * T * H * W, E]),
                    noise.reshape([B * T * H * W, E]),
                    top_logits.reshape([B * T * H * W, -1]),
                )
            else:
                raise NotImplementedError
            loss = (
                self.importance_weight * importance_loss + self.load_weight * load_loss
            )
        else:
            loss = None

        return routing_weights, top_k_gates, top_k_indices, loss


class LinearGatingNet(GatingNet):
    def __init__(self, moe_config, input_shape, in_channels):
        super().__init__(moe_config, input_shape, in_channels)
        assert len(input_shape) == 4
        T, H, W, C = input_shape

        self.lin = nn.Linear(
            in_features=in_channels, out_features=self.num_experts, bias_attr=False
        )

    def gating(self, x, t_map=None):
        routing_weights = self.lin(x)  # [B, T, H, W, E]
        return routing_weights


class SpatialLatentGatingNet(GatingNet):
    def __init__(self, moe_config, input_shape, in_channels):
        super().__init__(moe_config, input_shape, in_channels)
        assert len(input_shape) == 4
        T, H, W, C = input_shape

        gain = 1.0
        fan = self.out_planes / self.num_experts
        bound = gain * math.sqrt(3.0 / fan)
        self.routing_weights = paddle.create_parameter(
            shape=[H, W, self.num_experts],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-bound, bound),
        )

    def gating(self, x, t_map=None):
        # assert t_map is not None
        routing_weights = self.routing_weights.unsqueeze(0).tile(
            [x.shape[0], x.shape[1], 1, 1, 1]
        )  # [B, T, H, W, E]
        return routing_weights


class SpatialLatentLinearGatingNet(GatingNet):
    def __init__(self, moe_config, input_shape, in_channels):
        super().__init__(moe_config, input_shape, in_channels)
        assert len(input_shape) == 4
        T, H, W, C = input_shape

        gain = 1.0
        fan = self.out_planes / self.num_experts
        bound = gain * math.sqrt(3.0 / fan)
        self.spatial_routing_weights = paddle.create_parameter(
            shape=[H, W, self.num_experts],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-bound, bound),
        )
        self.lin = nn.Linear(
            in_features=in_channels, out_features=self.num_experts, bias_attr=False
        )

        self.combine_weight = paddle.create_parameter(
            shape=[H, W, self.num_experts, 2],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-bound, bound),
        )

    def gating(self, x, t_map=None):
        # assert t_map is not None
        spatial_routing_weights = self.spatial_routing_weights.tile(
            [x.shape[0], x.shape[1], 1, 1, 1]
        )  # [B, T, H, W, E]
        linear_routing_weights = self.lin(x)  # [B, T, H, W, E]
        routing_weights = paddle.stack(
            [spatial_routing_weights, linear_routing_weights], axis=-1
        )  # [B, T, H, W, E, 2]
        combine_weight = self.combine_weight.tile(
            [x.shape[0], x.shape[1], 1, 1, 1, 1]
        )  # [B, T, H, W, E, 2]
        routing_weights = (routing_weights * combine_weight).sum(-1)  # [B, T, H, W, E]
        return routing_weights


class CuboidLatentGatingNet(GatingNet):
    def __init__(self, moe_config, input_shape, in_channels):
        super().__init__(moe_config, input_shape, in_channels)
        assert len(input_shape) == 4
        T, H, W, C = input_shape

        gain = 1.0
        fan = self.out_planes / self.num_experts
        bound = gain * math.sqrt(3.0 / fan)
        self.routing_weights = paddle.create_parameter(
            shape=[T, H, W, self.num_experts],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-bound, bound),
        )

    def gating(self, x, t_map=None):
        # assert t_map is not None
        routing_weights = self.routing_weights.unsqueeze(0).tile(
            [x.shape[0], 1, 1, 1, 1]
        )  # [B, T, H, W, E]
        return routing_weights


class CuboidLatentLinearGatingNet(GatingNet):
    def __init__(self, moe_config, input_shape, in_channels):
        super().__init__(moe_config, input_shape, in_channels)
        assert len(input_shape) == 4
        T, H, W, C = input_shape

        gain = 1.0
        fan = self.out_planes / self.num_experts
        bound = gain * math.sqrt(3.0 / fan)
        self.cuboid_routing_weights = paddle.create_parameter(
            shape=[T, H, W, self.num_experts],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-bound, bound),
        )

        self.lin = nn.Linear(
            in_features=in_channels, out_features=self.num_experts, bias_attr=False
        )

        self.combine_weight = paddle.create_parameter(
            shape=[T, H, W, self.num_experts, 2],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-bound, bound),
        )

    def gating(self, x, t_map=None):
        # assert t_map is not None
        cuboid_routing_weights = self.cuboid_routing_weights.unsqueeze(0).tile(
            [x.shape[0], 1, 1, 1, 1]
        )  # [B, T, H, W, E]
        linear_routing_weights = self.lin(x)  # [B, T, H, W, E]
        routing_weights = paddle.stack(
            [cuboid_routing_weights, linear_routing_weights], axis=-1
        )  # [B, T, H, W, E, 2]
        combine_weight = self.combine_weight.tile(
            [x.shape[0], 1, 1, 1, 1, 1]
        )  # [B, T, H, W, E, 2]
        routing_weights = (routing_weights * combine_weight).sum(-1)  # [B, T, H, W, E]
        return routing_weights


def aggregate_aux_losses(net):
    aux_losses = []
    for module in net.sublayers():
        if hasattr(module, "aux_loss"):
            aux_losses.append(module.aux_loss.unsqueeze(0))
    return aux_losses


# MoE Routing


class SparseDispatcherScatter(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = paddle.nonzero(gates).sort(
            0
        ), paddle.nonzero(gates).argsort(0)
        _, self._expert_index = sorted_experts.split(1, axis=1)
        self._batch_index = paddle.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = paddle.take_along_axis(
            gates_exp, axis=1, indices=self._expert_index
        )

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return paddle.split(inp_exp, self._part_sizes, axis=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = paddle.concat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.multiply(self._nonzero_gates)
        zeros = paddle.zeros([self._gates.shape[0], expert_out[-1].shape[1]])
        zeros.stop_gradient = False
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


class SparseDispatcher(object):
    def __init__(self, num_experts, top_k_gates, top_k_indices):
        self.num_experts = num_experts
        self.gates = top_k_gates  # [B, K]
        self.gate_inds = top_k_indices  # [B, K]
        E = num_experts
        B, K = top_k_gates.shape
        self.batch_index_per_expert = paddle.stack(
            [
                (top_k_indices == expert_id).sum(-1).astype("bool")
                for expert_id in range(E)
            ],
            axis=0,
        )  # [E, B]
        self.gates_per_expert = paddle.concat(
            [top_k_gates[top_k_indices == expert_id] for expert_id in range(E)]
        )  # B * K
        self.batch_index_all = paddle.nonzero(self.batch_index_per_expert)[
            :, 1
        ]  # B * K
        self.expert_size = self.batch_index_per_expert.sum(-1)  # [E]

    def dispatch(self, x):
        B, C = x.shape
        dispatched_x = [
            x[batch_index] for batch_index in self.batch_index_per_expert
        ]  # E * [B_e, C]
        return dispatched_x

    def combine(self, expert_out):
        # expert_out: E * [B_e, C]
        assert len(expert_out) == self.num_experts
        com_res = paddle.concat(expert_out, axis=0)  # [B * K, C]
        zeros = paddle.zeros([self.gates.shape[0], com_res.shape[1]])
        zeros.stop_gradient = False
        combined_res = zeros.index_add(
            axis=0,
            index=self.batch_index_all,
            value=com_res * self.gates_per_expert.unsqueeze(-1),
        )
        return combined_res


class DenseDispatcher(object):
    def __init__(self, num_experts, top_k_gates, top_k_indices):
        self.num_experts = num_experts
        self.gates = top_k_gates  # [B, K]
        self.gate_inds = top_k_indices  # [B, K]

    def combine(self, expert_out):
        # expert_out: [B, E, C]
        B, E, C = expert_out.shape
        assert E == self.num_experts
        selected_out = paddle.take_along_axis(
            expert_out, axis=1, indices=self.gate_inds.unsqueeze(-1)
        )  # [B, K, C]
        combined_res = (selected_out * self.gates.unsqueeze(-1)).sum(1)
        return combined_res


# RNC


class LabelDifference(nn.Layer):
    def __init__(self, distance_type="l1"):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        assert labels.ndim == 3
        if self.distance_type == "l1":
            return paddle.abs(labels[:, :, None, :] - labels[:, None, :, :]).sum(
                axis=-1
            )
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Layer):
    def __init__(self, similarity_type="l2", temperature=2):
        super().__init__()
        self.similarity_type = similarity_type
        self.t = temperature

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        assert features.ndim == 3
        if self.similarity_type == "l2":
            logits = -(features[:, :, None, :] - features[:, None, :, :]).norm(
                2, axis=-1
            )
            logits /= self.t
            logits_max = paddle.max(logits, axis=1, keepdim=True)
            logits -= logits_max.detach()
            return logits
        elif self.similarity_type == "cosine":
            cos_func = nn.CosineSimilarity(axis=-1)
            logits = cos_func(features[:, :, None, :], features[:, None, :, :])
            logits /= self.t
            return logits
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Layer):
    def __init__(self, rnc_config):
        super().__init__()

        self.rank_mode = rnc_config["rank_imbalance_style"]
        self.t = rnc_config["rank_imbalance_temp"]
        self.label_diff_fn = LabelDifference(rnc_config["label_difference_style"])
        self.feature_sim_fn = FeatureSimilarity(
            rnc_config["feature_similarity_style"], self.t
        )
        self.rnc_weight = rnc_config["rank_reg_coeff"]
        self.loss_cal_mode = rnc_config["loss_cal_style"]
        self.softmax_cri = nn.Softmax(axis=-1)

    def cal_loss(self, features, labels):

        B = features.shape[0]
        assert B > 1
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features)
        exp_logits = logits.exp()
        n = logits.shape[1]

        # remove diagonal
        logits = logits.masked_select(
            (1 - paddle.eye(n)).astype("bool").unsqueeze(0).tile([B, 1, 1])
        ).reshape([B, n, n - 1])
        exp_logits = exp_logits.masked_select(
            (1 - paddle.eye(n)).astype("bool").unsqueeze(0).tile([B, 1, 1])
        ).reshape([B, n, n - 1])
        label_diffs = label_diffs.masked_select(
            (1 - paddle.eye(n)).astype("bool").unsqueeze(0).tile([B, 1, 1])
        ).reshape([B, n, n - 1])

        if self.loss_cal_mode == "memory-efficient":
            loss = 0.0
            for k in range(n - 1):
                pos_logits = logits[:, :, k]  # [B, n]
                pos_label_diffs = label_diffs[:, :, k]  # [B, n]
                neg_mask = (label_diffs >= pos_label_diffs.unsqueeze(-1)).astype(
                    "float32"
                )  # [B, n, n - 1]
                pos_log_probs = pos_logits - paddle.log(
                    (neg_mask * exp_logits).sum(axis=-1)
                )  # [B, n]
                loss += -pos_log_probs.sum()
            loss /= B * n * (n - 1)
        elif self.loss_cal_mode == "computation-efficient":
            neg_mask = (label_diffs.unsqueeze(-2) >= label_diffs.unsqueeze(-1)).astype(
                "float32"
            )  # [B, n, n - 1, n - 1]
            pos_log_probs = logits - paddle.log(
                (neg_mask * exp_logits.unsqueeze(-2).tile([1, 1, n - 1, 1])).sum(
                    axis=-1
                )
            )  # [B, n, n - 1]
            loss = -pos_log_probs.mean()
        else:
            raise NotImplementedError

        return loss

    def forward(self, features, labels):
        # features: [B, T_o, H, W, C_o]
        # labels: [B, T_o, H, W, C_l]

        B, T_o, H, W, C_o = features.shape
        _, _, _, _, C_l = labels.shape

        loss = None
        if self.rank_mode == "batch":
            features = features.reshape([B, -1, C_o]).transpose([1, 0, 2])
            labels = labels.reshape([B, -1, C_l]).transpose([1, 0, 2])
            loss = self.cal_loss(features, labels)
        elif self.rank_mode == "batch+T+H+W":
            feat = features.transpose([0, 2, 3, 1, 4]).reshape([-1, T_o, C_o])
            label = labels.transpose([0, 2, 3, 1, 4]).reshape([-1, T_o, C_l])
            loss_T = self.cal_loss(feat, label)

            feat = features.transpose([0, 1, 3, 2, 4]).reshape([-1, H, C_o])
            label = labels.transpose([0, 1, 3, 2, 4]).reshape([-1, H, C_l])
            loss_H = self.cal_loss(feat, label)

            feat = features.reshape([-1, W, C_o])
            label = labels.reshape([-1, W, C_l])
            loss_W = self.cal_loss(feat, label)

            feat = features.transpose([1, 2, 3, 0, 4]).reshape([-1, B, C_o])
            label = labels.transpose([1, 2, 3, 0, 4]).reshape([-1, B, C_l])
            loss_batch = self.cal_loss(feat, label)

            loss = loss_T + loss_H + loss_W + loss_batch
        else:
            raise NotImplementedError

        loss = self.rnc_weight * loss

        return loss
