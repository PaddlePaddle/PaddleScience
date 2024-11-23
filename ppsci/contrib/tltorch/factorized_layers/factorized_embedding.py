import numpy as np
import paddle
from paddle import nn

from ..factorized_tensors import TensorizedTensor
from ..factorized_tensors import tensor_init
from ..utils import get_tensorized_shape

# Authors: Cole Hawkins
#          Jean Kossaifi


class FactorizedEmbedding(nn.Layer):
    """
    Tensorized Embedding Layers For Efficient Model Compression
    Tensorized drop-in replacement for `torch.nn.Embedding`

    Parameters
    ----------
    num_embeddings : int
        number of entries in the lookup table
    embedding_dim : int
        number of dimensions per entry
    auto_tensorize : bool
        whether to use automatic reshaping for the embedding dimensions
    n_tensorized_modes : int or int tuple
        number of reshape dimensions for both embedding table dimension
    tensorized_num_embeddings : int tuple
        tensorized shape of the first embedding table dimension
    tensorized_embedding_dim : int tuple
        tensorized shape of the second embedding table dimension
    factorization : str
        tensor type
    rank : int tuple or str
        rank of the tensor factorization
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        auto_tensorize=True,
        n_tensorized_modes=3,
        tensorized_num_embeddings=None,
        tensorized_embedding_dim=None,
        factorization="blocktt",
        rank=8,
        n_layers=1,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if auto_tensorize:

            if (
                tensorized_num_embeddings is not None
                and tensorized_embedding_dim is not None
            ):
                raise ValueError(
                    "Either use auto_tensorize or specify tensorized_num_embeddings and tensorized_embedding_dim."
                )

            tensorized_num_embeddings, tensorized_embedding_dim = get_tensorized_shape(
                in_features=num_embeddings,
                out_features=embedding_dim,
                order=n_tensorized_modes,
                min_dim=2,
                verbose=False,
            )

        else:
            # check that dimensions match factorization
            computed_num_embeddings = np.prod(tensorized_num_embeddings)
            computed_embedding_dim = np.prod(tensorized_embedding_dim)

            if computed_num_embeddings != num_embeddings:
                raise ValueError(
                    "Tensorized embeddding number {} does not match num_embeddings argument {}".format(
                        computed_num_embeddings, num_embeddings
                    )
                )
            if computed_embedding_dim != embedding_dim:
                raise ValueError(
                    "Tensorized embeddding dimension {} does not match embedding_dim argument {}".format(
                        computed_embedding_dim, embedding_dim
                    )
                )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_shape = (tensorized_num_embeddings, tensorized_embedding_dim)
        self.weight_shape = (self.num_embeddings, self.embedding_dim)

        self.n_layers = n_layers
        if n_layers > 1:
            self.tensor_shape = (n_layers,) + self.tensor_shape
            self.weight_shape = (n_layers,) + self.weight_shape

        self.factorization = factorization

        self.weight = TensorizedTensor.new(
            self.tensor_shape,
            rank=rank,
            factorization=self.factorization,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

        self.rank = self.weight.rank

    def reset_parameters(self):
        # Parameter initialization from Yin et al.
        # TT-Rec: Tensor Train Compression for Deep Learning Recommendation Model Embeddings
        target_stddev = 1 / np.sqrt(3 * self.num_embeddings)
        with paddle.no_grad():
            tensor_init(self.weight, std=target_stddev)

    def forward(self, input, indices=0):
        # to handle case where input is not 1-D
        output_shape = (*input.shape, self.embedding_dim)

        flattened_input = input.reshape([-1])

        if self.n_layers == 1:
            if indices == 0:
                embeddings = self.weight[flattened_input, :]
        else:
            embeddings = self.weight[indices, flattened_input, :]

        # CPTensorized returns CPTensorized when indexing
        if self.factorization.lower() == "cp":
            embeddings = embeddings.to_matrix()

        # TuckerTensorized returns tensor not matrix,
        # and requires reshape not view for contiguous
        elif self.factorization.lower() == "tucker":
            embeddings = embeddings.reshape([input.shape[0], -1])

        return embeddings.view(output_shape)

    @classmethod
    def from_embedding(
        cls,
        embedding_layer,
        rank=8,
        factorization="blocktt",
        n_tensorized_modes=2,
        decompose_weights=True,
        auto_tensorize=True,
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        """
        Create a tensorized embedding layer from a regular embedding layer

        Parameters
        ----------
        embedding_layer : torch.nn.Embedding
        rank : int tuple or str
            rank of the tensor decomposition
        factorization : str
            tensor type
        decompose_weights: bool
            whether to decompose weights and use for initialization
        auto_tensorize: bool
            if True, automatically reshape dimensions for TensorizedTensor
        decomposition_kwargs: dict
            specify kwargs for the decomposition
        """
        num_embeddings, embedding_dim = embedding_layer.weight.shape

        instance = cls(
            num_embeddings,
            embedding_dim,
            auto_tensorize=auto_tensorize,
            factorization=factorization,
            n_tensorized_modes=n_tensorized_modes,
            rank=rank,
            **kwargs,
        )

        if decompose_weights:
            with paddle.no_grad():
                instance.weight.init_from_matrix(
                    embedding_layer.weight.data, **decomposition_kwargs
                )

        else:
            instance.reset_parameters()

        return instance

    @classmethod
    def from_embedding_list(
        cls,
        embedding_layer_list,
        rank=8,
        factorization="blocktt",
        n_tensorized_modes=2,
        decompose_weights=True,
        auto_tensorize=True,
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        """
        Create a tensorized embedding layer from a regular embedding layer

        Parameters
        ----------
        embedding_layer : torch.nn.Embedding
        rank : int tuple or str
            tensor rank
        factorization : str
            tensor decomposition to use
        decompose_weights: bool
            decompose weights and use for initialization
        auto_tensorize: bool
            automatically reshape dimensions for TensorizedTensor
        decomposition_kwargs: dict
            specify kwargs for the decomposition
        """
        n_layers = len(embedding_layer_list)
        num_embeddings, embedding_dim = embedding_layer_list[0].weight.shape

        for i, layer in enumerate(embedding_layer_list[1:]):
            # Just some checks on the size of the embeddings
            # They need to have the same size so they can be jointly factorized
            new_num_embeddings, new_embedding_dim = layer.weight.shape
            if num_embeddings != new_num_embeddings:
                msg = "All embedding layers must have the same num_embeddings."
                msg += f"Yet, got embedding_layer_list[0] with num_embeddings={num_embeddings} "
                msg += f" and embedding_layer_list[{i+1}] with num_embeddings={new_num_embeddings}."
                raise ValueError(msg)
            if embedding_dim != new_embedding_dim:
                msg = "All embedding layers must have the same embedding_dim."
                msg += f"Yet, got embedding_layer_list[0] with embedding_dim={embedding_dim} "
                msg += f" and embedding_layer_list[{i+1}] with embedding_dim={new_embedding_dim}."
                raise ValueError(msg)

        instance = cls(
            num_embeddings,
            embedding_dim,
            n_tensorized_modes=n_tensorized_modes,
            auto_tensorize=auto_tensorize,
            factorization=factorization,
            rank=rank,
            n_layers=n_layers,
            **kwargs,
        )

        if decompose_weights:
            weight_tensor = paddle.stack(
                [layer.weight.data for layer in embedding_layer_list]
            )
            with paddle.no_grad():
                instance.weight.init_from_matrix(weight_tensor, **decomposition_kwargs)

        else:
            instance.reset_parameters()

        return instance

    def get_embedding(self, indices):
        if self.n_layers == 1:
            raise ValueError(
                "A single linear is parametrized, directly use the main class."
            )

        return SubFactorizedEmbedding(self, indices)


class SubFactorizedEmbedding(nn.Layer):
    """Class representing one of the embeddings from the mother joint factorized embedding layer

    Parameters
    ----------

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """

    def __init__(self, main_layer, indices):
        super().__init__()
        self.main_layer = main_layer
        self.indices = indices

    def forward(self, x):
        return self.main_layer(x, self.indices)

    def extra_repr(self):
        return ""

    def __repr__(self):
        msg = f" {self.__class__.__name__} {self.indices} from main factorized layer."
        msg += f"\n{self.__class__.__name__}("
        msg += self.extra_repr()
        msg += ")"
        return msg
