import math
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import LayerNorm, Linear, Layer, ReLU, LayerList, ParameterList
from tqdm import tqdm

from paddle_geometric.explain import Explanation
from paddle_geometric.explain.algorithm import ExplainerAlgorithm
from paddle_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from paddle_geometric.nn import MessagePassing


def explain_message(self, out: Tensor, x_i: Tensor, x_j: Tensor) -> Tensor:
    basis_messages = F.layer_norm(out, normalized_shape=out.shape[-1:]).relu()

    if getattr(self, 'message_scale', None) is not None:
        basis_messages = basis_messages * self.message_scale.unsqueeze(-1)

        if self.message_replacement is not None:
            if basis_messages.shape == self.message_replacement.shape:
                basis_messages = (
                    basis_messages +
                    (1 - self.message_scale).unsqueeze(-1) * self.message_replacement
                )
            else:
                basis_messages = (
                    basis_messages +
                    ((1 - self.message_scale).unsqueeze(-1) *
                     self.message_replacement.unsqueeze(0))
                )

    self.latest_messages = basis_messages
    self.latest_source_embeddings = x_j
    self.latest_target_embeddings = x_i

    return basis_messages


class GraphMaskExplainer(ExplainerAlgorithm):
    coeffs = {
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(
        self,
        num_layers: int,
        epochs: int = 100,
        lr: float = 0.01,
        penalty_scaling: int = 5,
        lambda_optimizer_lr: float = 1e-2,
        init_lambda: float = 0.55,
        allowance: float = 0.03,
        allow_multiple_explanations: bool = False,
        log: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert 0 <= penalty_scaling <= 10
        assert 0 <= init_lambda <= 1
        assert 0 <= allowance <= 1

        self.num_layers = num_layers
        self.init_lambda = init_lambda
        self.lambda_optimizer_lr = lambda_optimizer_lr
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance
        self.allow_multiple_explanations = allow_multiple_explanations
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.coeffs.update(kwargs)

    def forward(
        self,
        model: Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:

        hard_node_mask = None

        if self.model_config.task_level == ModelTaskLevel.node:
            hard_node_mask, hard_edge_mask = self._get_hard_masks(
                model, index, edge_index, num_nodes=x.shape[0])
        self._train_explainer(model, x, edge_index, target=target, index=index,
                              **kwargs)
        node_mask = self._post_process_mask(self.node_feat_mask,
                                            hard_node_mask, apply_sigmoid=True)
        edge_mask = self._explain(model, index=index)
        edge_mask = edge_mask[:edge_index.shape[1]]

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _hard_concrete(
        self,
        input_element: Tensor,
        summarize_penalty: bool = True,
        beta: float = 1 / 3,
        gamma: float = -0.2,
        zeta: float = 1.2,
        loc_bias: int = 2,
        min_val: int = 0,
        max_val: int = 1,
        training: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        input_element = input_element + loc_bias

        if training:
            u = paddle.rand_like(input_element)
            s = F.sigmoid(
                (paddle.log(u) - paddle.log(1 - u) + input_element) / beta)

            penalty = F.sigmoid(input_element - beta * math.log(-gamma / zeta))
        else:
            s = F.sigmoid(input_element)
            penalty = paddle.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (zeta - gamma) + gamma

        clipped_s = paddle.clip(s, min=min_val, max=max_val)

        clip_value = (paddle.min(clipped_s) + paddle.max(clipped_s)) / 2
        hard_concrete = (clipped_s > clip_value).astype(paddle.float32)
        clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def _set_masks(
        self,
        i_dim: List[int],
        j_dim: List[int],
        h_dim: List[int],
        x: Tensor,
    ):
        r"""Sets the node masks and edge masks."""
        num_nodes, num_feat = x.shape
        std = 0.1
        self.feat_mask_type = self.explainer_config.node_mask_type

        if self.feat_mask_type == MaskType.attributes:
            self.node_feat_mask = self.create_parameter(
                shape=[num_nodes, num_feat], default_initializer=paddle.nn.initializer.Normal(std=std)
            )
        elif self.feat_mask_type == MaskType.object:
            self.node_feat_mask = self.create_parameter(
                shape=[num_nodes, 1], default_initializer=paddle.nn.initializer.Normal(std=std)
            )
        else:
            self.node_feat_mask = self.create_parameter(
                shape=[1, num_feat], default_initializer=paddle.nn.initializer.Normal(std=std)
            )

        baselines, self.gates, full_biases = [], LayerList(), []

        for v_dim, m_dim, h_dim in zip(i_dim, j_dim, h_dim):
            self.transform, self.layer_norm = [], []

            input_dims = [v_dim, m_dim, v_dim]
            for input_dim in input_dims:
                self.transform.append(
                    Linear(input_dim, h_dim, bias_attr=False)
                )
                self.layer_norm.append(LayerNorm(h_dim))

            self.transforms = LayerList(self.transform)
            self.layer_norms = LayerList(self.layer_norm)

            self.full_bias = self.create_parameter(
                shape=[h_dim], default_initializer=paddle.nn.initializer.Constant(0.0)
            )
            full_biases.append(self.full_bias)

            self.reset_parameters(input_dims, h_dim)

            self.non_linear = ReLU()
            self.output_layer = Linear(h_dim, 1)

            gate = [
                self.transforms, self.layer_norms, self.non_linear,
                self.output_layer
            ]
            self.gates.extend(gate)

            baseline = self.create_parameter(
                shape=[m_dim],
                default_initializer=paddle.nn.initializer.Uniform(low=-1.0 / math.sqrt(m_dim), high=1.0 / math.sqrt(m_dim))
            )
            baselines.append(baseline)

        self.full_biases = ParameterList(full_biases)
        self.baselines = ParameterList(baselines)

        for param in self.parameters():
            param.stop_gradient = True

    def _enable_layer(self, layer: int):
        r"""Enables the input layer's edge mask."""
        for d in range(layer * 4, (layer * 4) + 4):
            for param in self.gates[d].parameters():
                param.stop_gradient = False
        self.full_biases[layer].stop_gradient = False
        self.baselines[layer].stop_gradient = False

    def reset_parameters(self, input_dims: List[int], h_dim: int):
        r"""Resets all learnable parameters of the module."""
        fan_in = sum(input_dims)
        std = math.sqrt(2.0 / float(fan_in + h_dim))
        a = math.sqrt(3.0) * std

        for transform in self.transforms:
            paddle.nn.initializer.Uniform(low=-a, high=a)(transform.weight)

        paddle.nn.initializer.Constant(0.0)(self.full_bias)

        for layer_norm in self.layer_norms:
            layer_norm.reset_parameters()

    def _loss(self, y_hat: Tensor, y: Tensor, penalty: float) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        g = F.relu(loss - self.allowance).mean()
        f = penalty * self.penalty_scaling

        loss = f + F.softplus(self.lambda_op) * g

        m = F.sigmoid(self.node_feat_mask)
        node_feat_reduce = getattr(paddle, self.coeffs['node_feat_reduction'])
        loss += self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * paddle.log(m + self.coeffs['EPS']) - (
            1 - m) * paddle.log(1 - m + self.coeffs['EPS'])
        loss += self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _freeze_model(self, module: Layer):
        r"""Freezes the parameters of the original GNN model by disabling
        their gradients.
        """
        for param in module.parameters():
            param.stop_gradient = True

    def _set_flags(self, model: Layer):
        r"""Initializes the underlying explainer model's parameters for each
        layer of the original GNN model.
        """
        for module in model.sublayers():
            if isinstance(module, MessagePassing):
                module.explain_message = explain_message.__get__(
                    module, MessagePassing)
                module.explain = True

    def _inject_messages(
        self,
        model: Layer,
        message_scale: List[Tensor],
        message_replacement: paddle.nn.ParameterList,
        set: bool = False,
    ):
        r"""Injects the computed messages into each layer of the original GNN
        model.
        """
        i = 0
        for module in model.sublayers():
            if isinstance(module, MessagePassing):
                if not set:
                    module.message_scale = message_scale[i]
                    module.message_replacement = message_replacement[i]
                    i += 1
                else:
                    module.message_scale = None
                    module.message_replacement = None

    def _train_explainer(
        self,
        model: Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model."""
        if not isinstance(index, (Tensor, int)) and index is not None:
            raise ValueError("'index' parameter can only be a 'Tensor', 'integer', or 'None'.")

        self._freeze_model(model)
        self._set_flags(model)

        input_dims, output_dims = [], []
        for module in model.sublayers():
            if isinstance(module, MessagePassing):
                input_dims.append(module.in_channels)
                output_dims.append(module.out_channels)

        self._set_masks(input_dims, output_dims, output_dims, x)

        optimizer = paddle.optimizer.Adam(parameters=self.parameters(), learning_rate=self.lr)

        for layer in reversed(range(self.num_layers)):
            if self.log:
                pbar = tqdm(total=self.epochs)
                if self.model_config.task_level == ModelTaskLevel.node:
                    pbar.set_description(f"Train explainer for node(s) {index} with layer {layer}")
                elif self.model_config.task_level == ModelTaskLevel.edge:
                    pbar.set_description(f"Train explainer for edge-level task with layer {layer}")
                else:
                    pbar.set_description(f"Train explainer for graph {index} with layer {layer}")

            self._enable_layer(layer)
            for epoch in range(self.epochs):
                with paddle.no_grad():
                    model(x, edge_index, **kwargs)

                gates, total_penalty = [], 0
                latest_source_embeddings, latest_messages = [], []
                latest_target_embeddings = []

                for module in model.sublayers():
                    if isinstance(module, MessagePassing):
                        latest_source_embeddings.append(module.latest_source_embeddings)
                        latest_messages.append(module.latest_messages)
                        latest_target_embeddings.append(module.latest_target_embeddings)

                gate_input = [latest_source_embeddings, latest_messages, latest_target_embeddings]
                for i in range(self.num_layers):
                    output = self.full_biases[i]
                    for j, gate in enumerate(gate_input):
                        try:
                            partial = self.gates[i * 4][j](gate[i])
                        except Exception:
                            self._set_masks(output_dims, output_dims, output_dims, x)
                            partial = self.gates[i * 4][j](gate[i])
                        result = self.gates[(i * 4) + 1][j](partial)
                        output += result

                    relu_output = self.gates[(i * 4) + 2](output / len(gate_input))
                    sampling_weights = self.gates[(i * 4) + 3](relu_output).squeeze(-1)
                    sampling_weights, penalty = self._hard_concrete(sampling_weights)
                    gates.append(sampling_weights)
                    total_penalty += penalty

                self._inject_messages(model, gates, self.baselines)

                self.lambda_op = paddle.create_parameter(
                    shape=[], dtype="float32", default_initializer=paddle.nn.initializer.Constant(self.init_lambda)
                )
                optimizer_lambda = paddle.optimizer.RMSProp(
                    parameters=[self.lambda_op], learning_rate=self.lambda_optimizer_lr, centered=True
                )

                optimizer.clear_grad()
                optimizer_lambda.clear_grad()

                h = x * F.sigmoid(self.node_feat_mask)
                y_hat, y = model(x=h, edge_index=edge_index, **kwargs), target

                if self.model_config.task_level in [ModelTaskLevel.node, ModelTaskLevel.edge]:
                    if index is not None:
                        y_hat, y = y_hat[index], y[index]

                self._inject_messages(model, gates, self.baselines, set=True)

                loss = self._loss(y_hat, y, total_penalty)

                loss.backward()
                optimizer.step()
                optimizer_lambda.step()

                if self.lambda_op.numpy()[0] < -2:
                    self.lambda_op.set_value(paddle.full_like(self.lambda_op, -2))
                elif self.lambda_op.numpy()[0] > 30:
                    self.lambda_op.set_value(paddle.full_like(self.lambda_op, 30))

                if self.log:
                    pbar.update(1)

            if self.log:
                pbar.close()

    def _explain(
        self,
        model: Layer,
        *,
        index: Optional[Union[int, Tensor]] = None,
    ) -> Tensor:
        r"""Generates explanations for the original GNN model."""
        if not isinstance(index, (Tensor, int)) and index is not None:
            raise ValueError("'index' parameter can only be a 'Tensor', 'integer', or 'None'.")

        self._freeze_model(model)
        self._set_flags(model)

        with paddle.no_grad():
            latest_source_embeddings, latest_messages = [], []
            latest_target_embeddings = []

            for module in model.sublayers():
                if isinstance(module, MessagePassing):
                    latest_source_embeddings.append(module.latest_source_embeddings)
                    latest_messages.append(module.latest_messages)
                    latest_target_embeddings.append(module.latest_target_embeddings)

            gate_input = [latest_source_embeddings, latest_messages, latest_target_embeddings]
            if self.log:
                pbar = tqdm(total=self.num_layers)

            for i in range(self.num_layers):
                if self.log:
                    pbar.set_description("Explain")
                output = self.full_biases[i]
                for j, gate in enumerate(gate_input):
                    partial = self.gates[i * 4][j](gate[i])
                    result = self.gates[(i * 4) + 1][j](partial)
                    output += result
                relu_output = self.gates[(i * 4) + 2](output / len(gate_input))
                sampling_weights = self.gates[(i * 4) + 3](relu_output).squeeze(-1)
                sampling_weights, _ = self._hard_concrete(sampling_weights, training=False)

                if i == 0:
                    edge_weight = sampling_weights
                else:
                    edge_weight = paddle.concat([edge_weight, sampling_weights], axis=0)

                if self.log:
                    pbar.update(1)

        if self.log:
            pbar.close()

        edge_mask = edge_weight.reshape([-1, edge_weight.shape[0] // self.num_layers])
        edge_mask = paddle.mean(edge_mask, axis=0)

        return edge_mask