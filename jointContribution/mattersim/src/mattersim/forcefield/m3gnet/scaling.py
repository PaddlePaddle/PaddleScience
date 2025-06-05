"""
Atomic scaling module. Used for predicting extensive properties.
"""
from typing import Optional
from typing import Union

import numpy as np
import paddle
from ase import Atoms
from mattersim.datasets.utils.regressor import solver
from mattersim.utils.paddle_utils import scatter_mean

DATA_INDEX = {
    "total_energy": 0,
    "forces": 2,
    "per_atom_energy": 1,
    "per_species_energy": 0,
}


class AtomScaling(paddle.nn.Layer):
    """
    Atomic extensive property rescaling module
    """

    def __init__(
        self,
        atoms: list[Atoms] = None,
        total_energy: list[float] = None,
        forces: list[np.ndarray] = None,
        atomic_numbers: list[np.ndarray] = None,
        num_atoms: list[float] = None,
        max_z: int = 94,
        scale_key: str = None,
        shift_key: str = None,
        init_scale: Union[paddle.Tensor, float] = None,
        init_shift: Union[paddle.Tensor, float] = None,
        trainable_scale: bool = False,
        trainable_shift: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            forces: a list of atomic forces (np.ndarray) in each graph
            max_z: (int) maximum atomic number
                - if scale_key or shift_key is specified,
                  max_z should be equal to the maximum atomic_number.
            scale_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms (default)
            shift_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean :
                  default option is gaussian regression (NequIP)
                - per_species_energy_mean_linear_reg :
                  an alternative choice is linear regression (M3GNet)
            init_scale (paddle.Tensor or float)
            init_shift (paddle.Tensor or float)
        """
        super().__init__()
        self.max_z = max_z
        if scale_key or shift_key:
            total_energy = paddle.to_tensor(data=np.array(total_energy))
            forces = (
                paddle.to_tensor(data=np.concatenate(forces, axis=0))
                if forces is not None
                else None
            )
            if atomic_numbers is None:
                atomic_numbers = [atom.get_atomic_numbers() for atom in atoms]
            atomic_numbers = (
                paddle.to_tensor(data=np.concatenate(atomic_numbers, axis=0))
                .squeeze(axis=-1)
                .astype(dtype="int64")
            )
            if num_atoms is None:
                num_atoms = [atom.positions.shape[0] for atom in atoms]
            num_atoms = paddle.to_tensor(data=np.array(num_atoms))
            per_atom_energy = total_energy / num_atoms
            data_list = [total_energy, per_atom_energy, forces]
            assert (
                tuple(num_atoms.shape)[0] == tuple(total_energy.shape)[0]
            ), "num_atoms and total_energy should have the same size, "
            f"""but got {tuple(num_atoms.shape)[0]} and {tuple(total_energy.shape)[0]}"""
            if forces is not None:
                assert (
                    tuple(forces.shape)[0] == tuple(atomic_numbers.shape)[0]
                ), "forces and atomic_numbers should have the same length, "
                f"""but got {tuple(forces.shape)[0]} and {tuple(atomic_numbers.shape)[0]}"""
            if (
                scale_key == "per_species_energy_std"
                and shift_key == "per_species_energy_mean"
                and init_shift is None
                and init_scale is None
            ):
                init_shift, init_scale = self.get_gaussian_statistics(
                    atomic_numbers, num_atoms, total_energy
                )
            else:
                if shift_key and init_shift is None:
                    init_shift = self.get_statistics(
                        shift_key, max_z, data_list, atomic_numbers, num_atoms
                    )
                if scale_key and init_scale is None:
                    init_scale = self.get_statistics(
                        scale_key, max_z, data_list, atomic_numbers, num_atoms
                    )
        if init_scale is None:
            init_scale = paddle.ones(shape=max_z + 1)
        elif isinstance(init_scale, float):
            init_scale = paddle.to_tensor(data=init_scale).tile(repeat_times=max_z + 1)
        else:
            assert tuple(init_scale.shape)[0] == max_z + 1
        if init_shift is None:
            init_shift = paddle.zeros(shape=max_z + 1)
        elif isinstance(init_shift, float):
            init_shift = paddle.to_tensor(data=init_shift).tile(repeat_times=max_z + 1)
        else:
            assert tuple(init_shift.shape)[0] == max_z + 1
        init_shift = init_shift.astype(dtype="float32")
        init_scale = init_scale.astype(dtype="float32")
        if trainable_scale is True:
            self.scale = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=init_scale
            )
        else:
            self.register_buffer(name="scale", tensor=init_scale)
        if trainable_shift is True:
            self.shift = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=init_shift
            )
        else:
            self.register_buffer(name="shift", tensor=init_shift)
        if verbose is True:
            print("Current scale: ", init_scale)
            print("Current shift: ", init_shift)

    def transform(
        self, atomic_energies: paddle.Tensor, atomic_numbers: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Take the origin values from model and get the transformed values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        normalized_energies = curr_scale * atomic_energies + curr_shift
        return normalized_energies

    def inverse_transform(
        self, atomic_energies: paddle.Tensor, atomic_numbers: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Take the transformed values and get the original values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        unnormalized_energies = (atomic_energies - curr_shift) / curr_scale
        return unnormalized_energies

    def forward(
        self, atomic_energies: paddle.Tensor, atomic_numbers: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Atomic_energies and atomic_numbers should have the same size
        """
        return self.transform(atomic_energies, atomic_numbers)

    def get_statistics(
        self, key, max_z, data_list, atomic_numbers, num_atoms
    ) -> paddle.Tensor:
        """
        Valid key:
            scale_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean
                - per_species_energy_mean_linear_reg :
                  an alternative choice is linear regression
            shift_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms
        """
        data = None
        for data_key in DATA_INDEX:
            if data_key in key:
                data = data_list[DATA_INDEX[data_key]]
        assert data is not None
        statistics = None
        if "mean" in key:
            if "per_species" in key:
                n_atoms = paddle.repeat_interleave(
                    paddle.arange(0, num_atoms.numel()), repeats=num_atoms
                )
                if "linear_reg" in key:
                    features = bincount(
                        atomic_numbers, n_atoms, minlength=self.max_z + 1
                    ).numpy()
                    data = data.numpy()
                    assert features.ndim == 2
                    features = features[(features > 0).any(axis=1)]
                    statistics = np.linalg.pinv(features.T.dot(y=features)).dot(
                        features.T.dot(y=data)
                    )
                    statistics = paddle.to_tensor(data=statistics)
                else:
                    N = bincount(atomic_numbers, num_atoms, minlength=self.max_z + 1)
                    assert N.ndim == 2
                    N = N[(N > 0).astype("bool").any(axis=1)]
                    N = N.astype(paddle.get_default_dtype())
                    statistics, _ = solver(
                        N, data, regressor="NormalizedGaussianProcess"
                    )
            else:
                statistics = paddle.mean(x=data).item()
        elif "std" in key:
            if "per_species" in key:
                print(
                    "Warning: calculating per_species_energy_std for full periodic table systems is risky, please use per_species_forces_rms instead."
                )
                n_atoms = paddle.repeat_interleave(
                    paddle.arange(0, num_atoms.numel(0)), repeats=num_atoms
                )
                N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
                assert N.ndim == 2
                N = N[(N > 0).astype("bool").any(axis=1)]
                N = N.astype(paddle.get_default_dtype())
                _, statistics = solver(N, data, regressor="NormalizedGaussianProcess")
            else:
                statistics = paddle.std(x=data).item()
        elif "rms" in key:
            if "per_species" in key:
                square = scatter_mean(
                    data.square(), atomic_numbers, dim=0, dim_size=max_z + 1
                )
                statistics = square.mean(axis=-1)
            else:
                statistics = paddle.sqrt(x=paddle.mean(x=data.square())).item()
        if isinstance(statistics, paddle.Tensor) is not True:
            statistics = paddle.to_tensor(data=statistics).tile(repeat_times=max_z + 1)
        assert tuple(statistics.shape)[0] == max_z + 1
        return statistics

    def get_gaussian_statistics(
        self,
        atomic_numbers: paddle.Tensor,
        num_atoms: paddle.Tensor,
        total_energy: paddle.Tensor,
    ):
        """
        Get the gaussian process mean and variance
        """
        n_atoms = paddle.repeat_interleave(
            paddle.arange(0, num_atoms.numel()), repeats=num_atoms
        )
        N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
        assert N.ndim == 2
        N = N[(N > 0).astype("bool").any(axis=1)]
        N = N.astype(paddle.get_default_dtype())
        mean, std = solver(N, total_energy, regressor="NormalizedGaussianProcess")
        assert tuple(mean.shape)[0] == self.max_z + 1
        assert tuple(std.shape)[0] == self.max_z + 1
        return mean, std


def bincount(
    input: paddle.Tensor, batch: Optional[paddle.Tensor] = None, minlength: int = 0
):
    assert input.ndim == 1
    if batch is None:
        return paddle.bincount(x=input, minlength=minlength)
    else:
        assert tuple(batch.shape) == tuple(input.shape)
        length = input.max_func().item() + 1
        if minlength == 0:
            minlength = length
        if length > minlength:
            raise ValueError(
                f"minlength {minlength} too small for input with integers up to and including {length}"
            )
        input_ = input + batch * minlength
        num_batch = batch.max_func() + 1
        return paddle.bincount(x=input_, minlength=minlength * num_batch).reshape(
            num_batch, minlength
        )
