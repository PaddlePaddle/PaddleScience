from __future__ import annotations

import contextlib
import inspect
import io
import pickle
import sys
import warnings
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from ase import Atoms
from ase import units
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import all_properties
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nptberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS
from ase.optimize.lbfgs import LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS
from ase.optimize.sciopt import SciPyFminCG
from chgnet.model.model import CHGNet
from chgnet.utils import determine_device
from pymatgen.analysis.eos import BirchMurnaghan
from pymatgen.core.structure import Molecule
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer
    from typing_extensions import Self
OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class CHGNetCalculator(Calculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = "energy", "forces", "stress", "magmoms"

    def __init__(
        self,
        model: (CHGNet | None) = None,
        *,
        use_device: (str | None) = None,
        check_cuda_mem: bool = False,
        stress_weight: (float | None) = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> None:
        """Provide a CHGNet instance to calculate various atomic properties using ASE.

        Args:
            model (CHGNet): instance of a chgnet model. If set to None,
                the pretrained CHGNet is loaded.
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        self.device = device
        if model is None:
            self.model = CHGNet.load(verbose=False, use_device=self.device)
        else:
            self.model = model.to(self.device)
        self.model.graph_converter.set_isolated_atom_response(on_isolated_atoms)
        self.stress_weight = stress_weight
        print(f"CHGNet will run on {self.device}")

    @classmethod
    def from_file(cls, path: str, use_device: (str | None) = None, **kwargs) -> Self:
        """Load a user's CHGNet model and initialize the Calculator."""
        return cls(model=CHGNet.from_file(path), use_device=use_device, **kwargs)

    @property
    def version(self) -> (str | None):
        """The version of CHGNet."""
        return self.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.model.n_params

    def calculate(
        self,
        atoms: (Atoms | None) = None,
        properties: (list | None) = None,
        system_changes: (list | None) = None,
    ) -> None:
        """Calculate various properties of the atoms using CHGNet.

        Args:
            atoms (Atoms | None): The atoms object to calculate properties for.
            properties (list | None): The properties to calculate.
                Default is all properties.
            system_changes (list | None): The changes made to the system.
                Default is all changes.
        """
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure)
        model_prediction = self.model.predict_graph(
            graph.to(self.device), task="efsm", return_crystal_feas=True
        )
        factor = 1 if not self.model.is_intensive else structure.composition.num_atoms
        self.results.update(
            energy=model_prediction["e"] * factor,
            forces=model_prediction["f"],
            free_energy=model_prediction["e"] * factor,
            magmoms=model_prediction["m"],
            stress=model_prediction["s"] * self.stress_weight,
            crystal_fea=model_prediction["crystal_fea"],
        )


class StructOptimizer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model: (CHGNet | CHGNetCalculator | None) = None,
        optimizer_class: (Optimizer | str | None) = "FIRE",
        use_device: (str | None) = None,
        stress_weight: float = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
        """
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )
        self.optimizer_class: Optimizer = optimizer_class
        if isinstance(model, CHGNetCalculator):
            self.calculator = model
        else:
            self.calculator = CHGNetCalculator(
                model=model,
                stress_weight=stress_weight,
                use_device=use_device,
                on_isolated_atoms=on_isolated_atoms,
            )

    @property
    def version(self) -> str:
        """The version of CHGNet."""
        return self.calculator.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.calculator.model.n_params

    def relax(
        self,
        atoms: (Structure | Atoms),
        *,
        fmax: (float | None) = 0.1,
        steps: (int | None) = 500,
        relax_cell: (bool | None) = True,
        ase_filter: (str | None) = "FrechetCellFilter",
        save_path: (str | None) = None,
        loginterval: (int | None) = 1,
        crystal_feas_save_path: (str | None) = None,
        verbose: bool = True,
        assign_magmoms: bool = True,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            ase_filter (str | ase.filters.Filter): The filter to apply to the atoms
                object for relaxation. Default = FrechetCellFilter
                Default used to be ExpCellFilter which was removed due to bug reported
                in https://gitlab.com/ase/ase/-/issues/1321 and fixed in
                https://gitlab.com/ase/ase/-/merge_requests/3024.
            save_path (str | None): The path to save the trajectory.
                Default = None
            loginterval (int | None): Interval for logging trajectory and crystal
                features. Default = 1
            crystal_feas_save_path (str | None): Path to save crystal feature vectors
                which are logged at a loginterval rage
                Default = None
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            assign_magmoms (bool): Whether to assign magnetic moments to the final
                structure. Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        from ase import filters
        from ase.filters import Filter

        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filters, inspect.isclass)
            if issubclass(cls, Filter)
        ]
        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filters, ase_filter)
            else:
                raise ValueError(
                    f"Invalid ase_filter={ase_filter!r}, must be one of {valid_filter_names}. "
                )
        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        atoms.calc = self.calculator
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)
            if relax_cell:
                atoms = ase_filter(atoms)
            optimizer: Optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)
            if crystal_feas_save_path:
                optimizer.attach(cry_obs, interval=loginterval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if save_path is not None:
            obs.save(save_path)
        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)
        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        if assign_magmoms:
            for key in struct.site_properties:
                struct.remove_site_property(property_name=key)
            struct.add_site_property(
                "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
            )
        return {"final_structure": struct, "trajectory": obs}


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    """CrystalFeasObserver is a hook in the relaxation and MD process that saves the
    intermediate crystal feature structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: (Atoms | Structure),
        *,
        model: (CHGNet | CHGNetCalculator | None) = None,
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        temperature: int = 300,
        starting_temperature: (int | None) = None,
        timestep: float = 2.0,
        pressure: float = 0.000101325,
        taut: (float | None) = None,
        taup: (float | None) = None,
        bulk_modulus: (float | None) = None,
        trajectory: (str | Trajectory | None) = None,
        logfile: (str | None) = None,
        loginterval: int = 1,
        crystal_feas_logfile: (str | None) = None,
        append_trajectory: bool = False,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        use_device: (str | None) = None,
    ) -> None:
        """Initialize the MD class.

        Args:
            atoms (Atoms): atoms to run the MD
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            ensemble (str): choose from 'nve', 'nvt', 'npt'
                Default = "nvt"
            thermostat (str): Thermostat to use
                choose from "Nose-Hoover", "Berendsen", "Berendsen_inhomogeneous"
                Default = "Berendsen_inhomogeneous"
            temperature (float): temperature for MD simulation, in K
                Default = 300
            starting_temperature (float): starting temperature of MD simulation, in K
                if set as None, the MD starts with the momentum carried by ase.Atoms
                if input is a pymatgen.core.Structure, the MD starts at 0K
                Default = None
            timestep (float): time step in fs
                Default = 2
            pressure (float): pressure in GPa
                Can be 3x3 or 6 np.array if thermostat is "Nose-Hoover"
                Default = 1.01325e-4 GPa = 1 atm
            taut (float): time constant for temperature coupling in fs.
                The temperature will be raised to target temperature in approximate
                10 * taut time.
                Default = 100 * timestep
            taup (float): time constant for pressure coupling in fs
                Default = 1000 * timestep
            bulk_modulus (float): bulk modulus of the material in GPa.
                Used in NPT ensemble for the barostat pressure coupling.
                The DFT bulk modulus can be found for most materials at
                https://next-gen.materialsproject.org/

                In NPT ensemble, the effective damping time for pressure is multiplied
                by compressibility. In LAMMPS, Bulk modulus is defaulted to 10
                see: https://docs.lammps.org/fix_press_berendsen.html
                and: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py

                If bulk modulus is not provided here, it will be calculated by CHGNet
                through Birch Murnaghan equation of state (EOS).
                Note the EOS fitting can fail because of non-parabolic potential
                energy surface, which is common with soft system like liquid and gas.
                In such case, user should provide an input bulk modulus for better
                barostat coupling, otherwise a guessed bulk modulus = 2 GPa will be used
                (water's bulk modulus)

                Default = None
            trajectory (str or Trajectory): Attach trajectory object
                Default = None
            logfile (str): open this file for recording MD outputs
                Default = None
            loginterval (int): write to log file every interval steps
                Default = 1
            crystal_feas_logfile (str): open this file for recording crystal features
                during MD. Default = None
            append_trajectory (bool): Whether to append to prev trajectory.
                If false, previous trajectory gets overwritten
                Default = False
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            use_device (str): the device for the MD run
                Default = None
        """
        self.ensemble = ensemble
        self.thermostat = thermostat
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        if starting_temperature is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=starting_temperature, force_temp=True
            )
            Stationary(atoms)
        self.atoms = atoms
        if isinstance(model, CHGNetCalculator):
            self.atoms.calc = model
        else:
            self.atoms.calc = CHGNetCalculator(
                model=model, use_device=use_device, on_isolated_atoms=on_isolated_atoms
            )
        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep
        if ensemble.lower() == "nve":
            """
            VelocityVerlet (constant N, V, E) molecular dynamics.

            Note: it's recommended to use smaller timestep for NVE compared to other
            ensembles, since the VelocityVerlet algorithm assumes a strict conservative
            force field.
            """
            self.dyn = VelocityVerlet(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
            print("NVE-MD created")
        elif ensemble.lower() == "nvt":
            """
            Constant volume/temperature molecular dynamics.
            """
            if thermostat.lower() == "nose-hoover":
                """
                Nose-hoover (constant N, V, T) molecular dynamics.
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=None,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Nose-Hoover MD created")
            elif thermostat.lower().startswith("berendsen"):
                """
                Berendsen (constant N, V, T) molecular dynamics.
                """
                self.dyn = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut * units.fs,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', 'Berendsen_inhomogeneous'"
                )
        elif ensemble.lower() == "npt":
            """
            Constant pressure/temperature molecular dynamics.
            """
            if bulk_modulus is not None:
                bulk_modulus_au = bulk_modulus / 160.2176
                compressibility_au = 1 / bulk_modulus_au
            else:
                try:
                    eos = EquationOfState(model=self.atoms.calc)
                    eos.fit(atoms=atoms, steps=500, fmax=0.1, verbose=False)
                    bulk_modulus = eos.get_bulk_modulus(unit="GPa")
                    bulk_modulus_au = eos.get_bulk_modulus(unit="eV/A^3")
                    compressibility_au = eos.get_compressibility(unit="A^3/eV")
                    print(
                        f"Completed bulk modulus calculation: k = {bulk_modulus:.3}GPa, {bulk_modulus_au:.3}eV/A^3"
                    )
                except Exception:
                    bulk_modulus_au = 2 / 160.2176
                    compressibility_au = 1 / bulk_modulus_au
                    warnings.warn(
                        "Warning!!! Equation of State fitting failed, setting bulk modulus to 2 GPa. NPT simulation can proceed with incorrect pressure relaxation time.User input for bulk modulus is recommended.",
                        stacklevel=2,
                    )
            self.bulk_modulus = bulk_modulus
            if thermostat.lower() == "nose-hoover":
                """
                Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an
                NPT (or N,stress,T) ensemble.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/npt.py
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                ptime = taup * units.fs
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=bulk_modulus * units.GPa * ptime * ptime,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Nose-Hoover MD created")
            elif thermostat.lower() == "berendsen_inhomogeneous":
                """
                Inhomogeneous_NPTBerendsen thermo/barostat
                This is a more flexible scheme that fixes three angles of the unit
                cell but allows three lattice parameter to change independently.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """
                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                )
                print("NPT-Berendsen-inhomogeneous-MD created")
            elif thermostat.lower() == "npt_berendsen":
                """
                This is a similar scheme to the Inhomogeneous_NPTBerendsen.
                This is a less flexible scheme that fixes the shape of the
                cell - three angles are fixed and the ratios between the three
                lattice constants.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """
                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', 'Berendsen_inhomogeneous'"
                )
        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep
        self.crystal_feas_logfile = crystal_feas_logfile

    def run(self, steps: int) -> None:
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        if self.crystal_feas_logfile:
            obs = CrystalFeasObserver(self.atoms)
            self.dyn.attach(obs, interval=self.loginterval)
        self.dyn.run(steps)
        if self.crystal_feas_logfile:
            obs.save(self.crystal_feas_logfile)

    def set_atoms(self, atoms: Atoms) -> None:
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD
        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator

    def upper_triangular_cell(self, *, verbose: (bool | None) = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5
            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]
            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)


class EquationOfState:
    """Class to calculate equation of state."""

    def __init__(
        self,
        model: (CHGNet | CHGNetCalculator | None) = None,
        optimizer_class: (Optimizer | str | None) = "FIRE",
        use_device: (str | None) = None,
        stress_weight: float = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "error",
    ) -> None:
        """Initialize a structure optimizer object for calculation of bulk modulus.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'error'
        """
        self.relaxer = StructOptimizer(
            model=model,
            optimizer_class=optimizer_class,
            use_device=use_device,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
        )
        self.fitted = False

    def fit(
        self,
        atoms: (Structure | Atoms),
        *,
        n_points: int = 11,
        fmax: (float | None) = 0.1,
        steps: (int | None) = 500,
        verbose: (bool | None) = False,
        **kwargs,
    ) -> None:
        """Relax the Structure/Atoms and fit the Birch-Murnaghan equation of state.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            n_points (int): Number of structures used in fitting the equation of states
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = False
            **kwargs: Additional parameters for the optimizer.

        Returns:
            Bulk Modulus (float)
        """
        if isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor.get_structure(atoms)
        primitive_cell = atoms.get_primitive_structure()
        local_minima = self.relaxer.relax(
            primitive_cell,
            relax_cell=True,
            fmax=fmax,
            steps=steps,
            verbose=verbose,
            **kwargs,
        )
        volumes, energies = [], []
        for idx in np.linspace(-0.1, 0.1, n_points):
            structure_strained = local_minima["final_structure"].copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = self.relaxer.relax(
                structure_strained,
                relax_cell=False,
                fmax=fmax,
                steps=steps,
                verbose=verbose,
                **kwargs,
            )
            volumes.append(result["final_structure"].volume)
            energies.append(result["trajectory"].energies[-1])
        self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
        self.bm.fit()
        self.fitted = True

    def get_bulk_modulus(self, unit: str = "eV/A^3") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "eV/A^3" or "GPa"
                Default = "eV/A^3"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "eV/A^3":
            return self.bm.b0
        if unit == "GPa":
            return self.bm.b0_GPa
        raise NotImplementedError("unit has to be eV/A^3 or GPa")

    def get_compressibility(self, unit: str = "A^3/eV") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "A^3/eV",
            "GPa^-1" "Pa^-1" or "m^2/N"
                Default = "A^3/eV"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "A^3/eV":
            return 1 / self.bm.b0
        if unit == "GPa^-1":
            return 1 / self.bm.b0_GPa
        if unit in {"Pa^-1", "m^2/N"}:
            return 1 / (self.bm.b0_GPa * 1000000000.0)
        raise NotImplementedError("unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N")
