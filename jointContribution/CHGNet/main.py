from ase.io.trajectory import Trajectory
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import MolecularDynamics
from chgnet.model.model import CHGNet
from chgnet.utils import solve_charge_by_mag
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# This is the paddle version of chgnet, which heavily references
# https://github.com/CederGroupHub/chgnet.

# load the CHGNet model
chgnet = CHGNet.load()
structure = Structure.from_file("chgnet/mp-18767-LiMnO2.cif")

# predict the structure
prediction = chgnet.predict_structure(structure)


for key, unit in [
    ("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B"),
]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")


# structure optimizer
relaxer = StructOptimizer()
# Perturb the structure
structure.perturb(0.1)
# Relax the perturbed structure
result = relaxer.relax(structure, verbose=True)

print("Relaxed structure:\n")
print(result["final_structure"])

print(result["trajectory"].energies)


# run molecular dynamics
md = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    ensemble="nvt",
    temperature=1000,  # in k
    timestep=2,  # in fs
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
)
md.run(50)  # run a 0.1 ps MD simulation


traj = Trajectory("md_out.traj")
mag = traj[-1].get_magnetic_moments()

# get the non-charge-decorated structure
structure = AseAtomsAdaptor.get_structure(traj[-1])
print(structure)

# get the charge-decorated structure
struct_with_chg = solve_charge_by_mag(structure)
print(struct_with_chg)
