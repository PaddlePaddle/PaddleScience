import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor

from paddle_geometric.explain import ExplainerAlgorithm
from paddle_geometric.explain.config import ModelMode, ModelTaskLevel
from paddle_geometric.explain.explanation import Explanation
from paddle_geometric.utils import k_hop_subgraph
from paddle_geometric.utils._subgraph import get_num_hops


class PGMExplainer(ExplainerAlgorithm):
    r"""The PGMExplainer model from the `"PGMExplainer: Probabilistic
    Graphical Model Explanations  for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper.

    The generated :class:`~paddle_geometric.explain.Explanation` provides a
    :obj:`node_mask` and a :obj:`pgm_stats` tensor, which stores the
    :math:`p`-values of each node as calculated by the Chi-squared test.

    Args:
        feature_index (List): The indices of the perturbed features. If set
            to :obj:`None`, all features are perturbed. (default: :obj:`None`)
        perturb_mode (str, optional): The method to generate the variations in
            features. One of :obj:`"randint"`, :obj:`"mean"`, :obj:`"zero"`,
            :obj:`"max"` or :obj:`"uniform"`. (default: :obj:`"randint"`)
        perturbations_is_positive_only (bool, optional): If set to :obj:`True`,
            restrict perturbed values to be positive. (default: :obj:`False`)
        is_perturbation_scaled (bool, optional): If set to :obj:`True`, will
            normalize the range of the perturbed features.
            (default: :obj:`False`)
        num_samples (int, optional): The number of samples of perturbations
            used to test the significance of nodes to the prediction.
            (default: :obj:`100`)
        max_subgraph_size (int, optional): The maximum number of neighbors to
            consider for the explanation. (default: :obj:`None`)
        significance_threshold (float, optional): The statistical threshold
            (:math:`p`-value) for which a node is considered to have an effect
            on the prediction. (default: :obj:`0.05`)
        pred_threshold (float, optional): The buffer value (in range
            :obj:`[0, 1]`) to consider the output from a perturbed data to be
            different from the original. (default: :obj:`0.1`)
    """
    def __init__(
        self,
        feature_index: Optional[List] = None,
        perturbation_mode: str = "randint",
        perturbations_is_positive_only: bool = False,
        is_perturbation_scaled: bool = False,
        num_samples: int = 100,
        max_subgraph_size: Optional[int] = None,
        significance_threshold: float = 0.05,
        pred_threshold: float = 0.1,
    ):
        super().__init__()
        self.feature_index = feature_index
        self.perturbation_mode = perturbation_mode
        self.perturbations_is_positive_only = perturbations_is_positive_only
        self.is_perturbation_scaled = is_perturbation_scaled
        self.num_samples = num_samples
        self.max_subgraph_size = max_subgraph_size
        self.significance_threshold = significance_threshold
        self.pred_threshold = pred_threshold

    def _perturb_features_on_nodes(
        self,
        x: Tensor,
        index: Tensor,
    ) -> Tensor:
        r"""Perturbs feature matrix :obj:`x`.

        Args:
            x (paddle.to_tensor): The feature matrix.
            index (paddle.to_tensor): The indices of nodes to perturb.
        """
        x_perturb = x.detach().clone()
        perturb_array = x_perturb[index]
        epsilon = 0.05 * paddle.max(x, axis=0)

        if self.perturbation_mode == "randint":
            perturb_array = paddle.randint(high=2, shape=perturb_array.shape)
        elif self.perturbation_mode == "mean":
            perturb_array[:, self.feature_index] = paddle.mean(
                x[:, self.feature_index])
        elif self.perturbation_mode == "zero":
            perturb_array[:, self.feature_index] = 0
        elif self.perturbation_mode == "max":
            perturb_array[:, self.feature_index] = paddle.max(
                x[:, self.feature_index])
        elif self.perturbation_mode == "uniform":
            random_perturbations = paddle.rand(
                perturb_array.shape) * 2 * epsilon - epsilon
            perturb_array[:, self.feature_index] = perturb_array[
                self.feature_index] + random_perturbations
            perturb_array.clamp(min=0, max=paddle.max(x, axis=0))

        if self.is_perturbation_scaled:
            perturb_array = paddle.multiply(
                perturb_array, paddle.rand(shape=perturb_array.shape)) * 2

        x_perturb[index] = perturb_array.astype(x_perturb.dtype)

        return x_perturb

    def _batch_perturb_features_on_node(
        self,
        model: paddle.nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        indices_to_perturb: np.array,
        percentage: float = 50.,  # % time node gets perturbed
        **kwargs,
    ) -> Tensor:
        r"""Perturbs the node features of a batch of graphs for graph
        classification tasks.

        Args:
            model (paddle.nn.Layer: The GNN model.
            x (paddle.to_tensor): The node feature matrix
            edge_index (paddle.to_tensor): The edge indices.
            indices_to_perturb (np.array): The indices of nodes to perturb.
            percentage (float, optional): The percentage of times a node gets
                perturbed. (default: :obj:`50.`)
            **kwargs (optional): Additional arguments passed to
                :meth:`model.forward`.
        """
        pred_paddle = model(x, edge_index, **kwargs)
        soft_pred = paddle.nn.functional.softmax(pred_paddle, axis=1)
        pred_label = paddle.argmax(soft_pred, axis=1)
        num_nodes = x.shape[0]

        samples = []
        for _ in range(self.num_samples):
            x_perturb = x.detach().clone()

            seeds = np.random.randint(0, 100, size=len(indices_to_perturb))
            perturbed_node_indexes = indices_to_perturb[(seeds < percentage)]
            x_perturb = self._perturb_features_on_nodes(
                x=x_perturb,
                index=perturbed_node_indexes,
            )
            sample = np.zeros(num_nodes + 1)
            sample[perturbed_node_indexes] = 1

            pred_perturb_paddle = model(x_perturb, edge_index, **kwargs)
            soft_pred_perturb = paddle.nn.functional.softmax(pred_perturb_paddle,
                                              axis=1).squeeze()

            pred_change = paddle.max(soft_pred) - soft_pred_perturb[pred_label]

            sample[num_nodes] = pred_change
            samples.append(sample)

        samples = paddle.to_tensor(np.array(samples))
        if self.perturbations_is_positive_only:
            samples = paddle.abs(samples)

        top = int(self.num_samples / 8)
        top_idx = paddle.argsort(samples[:, num_nodes])[-top:]
        for i in range(self.num_samples):
            if i in top_idx:
                samples[i, num_nodes] = 1
            else:
                samples[i, num_nodes] = 0

        return samples

    def _explain_graph(
        self,
        model: paddle.nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        target=None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        r"""Generates explanations for graph classification tasks.

        Args:
            model (paddle.nn.Layer: The model to explain.
            x (paddle.to_tensor): The node features.
            edge_index (paddle.to_tensor): The edge indices of the input graph.
            target (paddle.to_tensor, optional): The predicted label from the
                model. (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :meth:`model.forward`.

        Returns:
            pgm_nodes (List): The neighbor nodes that are significant in the
                selected node's prediction.
            pgm_stats (paddle.to_tensor): The :math:`p`-values of all the nodes in
                the graph, ordered by node index.
        """
        import pandas as pd
        from pgmpy.estimators.CITests import chi_square

        num_nodes = x.shape[0]
        if not self.max_subgraph_size:
            self.max_subgraph_size = int(num_nodes / 20)

        samples = self._batch_perturb_features_on_node(
            indices_to_perturb=np.array(range(num_nodes)),
            x=x,
            model=model,
            edge_index=edge_index,
        )

        # note: the PC estimator is in the original code, ie. est= PC(data)
        # but as it does nothing it is not included here
        data = pd.DataFrame(np.array(samples.detach().cpu()))

        p_values = []
        for node in range(num_nodes):
            chi2, p, _ = chi_square(
                node, int(target.detach().cpu()), [], data, boolean=False,
                significance_level=self.significance_threshold)
            p_values.append(p)

        # the original code uses number_candidates_nodes = int(top_nodes * 4)
        # if we consider 'top nodes' to equate to max number of nodes
        # it seems more correct to limit number_candidates_nodes to this
        candidate_nodes = np.argpartition(
            p_values, self.max_subgraph_size)[0:self.max_subgraph_size]

        # Round 2
        samples = self._batch_perturb_features_on_node(
            indices_to_perturb=candidate_nodes, x=x, edge_index=edge_index,
            model=model, **kwargs)

        # note: the PC estimator is in the original code, ie. est= PC(data)
        # but as it does nothing it is not included here
        data = pd.DataFrame(np.array(samples.detach().cpu()))

        p_values = []
        dependent_nodes = []

        target = num_nodes
        for node in range(num_nodes):
            _, p, _ = chi_square(
                node, target, [], data, boolean=False,
                significance_level=self.significance_threshold)
            p_values.append(p)
            if p < self.significance_threshold:
                dependent_nodes.append(node)

        top_p = np.min((self.max_subgraph_size, num_nodes - 1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)

        node_mask = paddle.zeros(x.size(), dtype=paddle.int)
        node_mask[pgm_nodes] = 1
        pgm_stats = paddle.to_tensor(p_values)

        return node_mask, pgm_stats

    def _explain_node(
        self,
        model: paddle.nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        target: Tensor,
        index: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        r"""Generates explanations for node classification tasks.

        Args:
            model (paddle.nn.Layer: The model to explain.
            x (paddle.to_tensor): The node features.
            edge_index (paddle.to_tensor): The edge indices of the input graph.
            target (paddle.to_tensor): The predicted label from the model.
            index (int): The index of the node for which the explanations is
                generated.
            **kwargs (optional): Additional arguments passed to
                :meth:`model.forward`.

        Returns:
            node_mask (paddle.to_tensor): A hard node mask corresponding to whether
                a node is significant in the selected node's prediction.
            pgm_stats (paddle.to_tensor): The :math:`p`-values of all the nodes in
                the graph, ordered by node index.
        """
        import pandas as pd
        from pgmpy.estimators.CITests import chi_square

        neighbors, _, _, _ = k_hop_subgraph(
            node_idx=index,
            num_hops=get_num_hops(model),
            edge_index=edge_index,
            relabel_nodes=False,
            num_nodes=x.shape[0],
        )

        if index not in neighbors:
            neighbors = paddle.concat([neighbors, index], axis=1)

        pred_model = model(x, edge_index, **kwargs)

        softmax_pred = paddle.nn.functional.softmax(pred_model, axis=1)

        samples = []
        pred_samples = []

        for _ in range(self.num_samples):
            # A subset of neighbors are selected randomly for perturbing:
            seeds = np.random.choice([1, 0], size=(len(neighbors), ))
            x_perturb = self._perturb_features_on_nodes(
                x=x,
                index=neighbors[seeds == 1],
            )

            # prediction after perturbation
            pred_perturb = model(x_perturb, edge_index, **kwargs)
            softmax_pred_perturb = paddle.nn.functional.softmax(pred_perturb, axis=1)
            sample_bool = np.ones(shape=(len(neighbors), ))
            sample_bool[((softmax_pred_perturb[neighbors, target] +
                          self.pred_threshold)
                         >= softmax_pred[neighbors, target]).cpu()] = 0

            samples.append(seeds)
            pred_samples.append(sample_bool)

        samples = np.asarray(samples)
        pred_samples = np.asarray(pred_samples)
        combine_samples = (samples * 10 + pred_samples) + 1

        neighbors = np.array(neighbors.detach().cpu())
        data_pgm = pd.DataFrame(combine_samples)
        data_pgm = data_pgm.rename(columns={
            0: "A",
            1: "B"
        })  # Trick to use chi_square test on first two data columns
        index_original_to_subgraph = dict(
            zip(neighbors, list(data_pgm.columns)))
        index_subgraph_to_original = dict(
            zip(list(data_pgm.columns), neighbors))
        p_values = []

        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:
            if node == index:
                # null hypothesis is perturbing a particular
                # node has no effect on result
                p = 0
            else:
                _, p, _ = chi_square(
                    index_original_to_subgraph[node],
                    index_original_to_subgraph[index], [], data_pgm,
                    boolean=False,
                    significance_level=self.significance_threshold)
            p_values.append(p)
            if p < self.significance_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        pgm_stats = paddle.ones(x.shape[0], dtype=paddle.float32)
        node_mask = paddle.zeros(x.shape, dtype=paddle.int32)

        pgm_stats[neighbors] = paddle.to_tensor(p_values, dtype=paddle.float32)

        if self.max_subgraph_size is None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((self.max_subgraph_size, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [
                index_subgraph_to_original[node] for node in ind_top_p
            ]
        node_mask[pgm_nodes] = 1
        return node_mask, pgm_stats

    def forward(
        self,
        model: paddle.nn.Layer,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,  # node index
        **kwargs,
    ) -> Explanation:

        if self.feature_index is None:
            self.feature_index = list(range(x.shape[-1]))

        if isinstance(index, Tensor):
            if index.numel() > 1:
                raise NotImplementedError(
                    f"'{self.__class__.__name}' only supports a single "
                    f"`index` for now")
            index = index.item()

        if self.model_config.task_level == ModelTaskLevel.node:
            node_mask, pgm_stats = self._explain_node(
                model=model,
                x=x,
                edge_index=edge_index,
                target=target[index],
                index=index,
                **kwargs,
            )
            return Explanation(
                x=x,
                edge_index=edge_index,
                node_mask=node_mask,
                pgm_stats=pgm_stats,
            )

        elif self.model_config.task_level == ModelTaskLevel.graph:
            node_mask, pgm_stats = self._explain_graph(
                model=model,
                x=x,
                target=target,
                edge_index=edge_index,
                **kwargs,
            )
            return Explanation(
                node_mask=node_mask,
                pgm_stats=pgm_stats,
            )

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False
        if self.explainer_config.edge_mask_type is not None:
            logging.error("Generation of edge masks is not supported")
            return False
        if self.model_config.mode == ModelMode.regression:
            logging.error("'PGMExplainer' only supports classification tasks")
            return False
        return True
