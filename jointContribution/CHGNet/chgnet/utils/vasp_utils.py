from __future__ import annotations

import os
import re
import warnings
from typing import TYPE_CHECKING

from chgnet.utils import write_json
from monty.io import zopen
from monty.os.path import zpath
from pymatgen.io.vasp.outputs import Oszicar
from pymatgen.io.vasp.outputs import Vasprun

if TYPE_CHECKING:
    from pymatgen.core import Structure


def parse_vasp_dir(
    base_dir: str,
    *,
    check_electronic_convergence: bool = True,
    save_path: (str | None) = None,
) -> dict[str, list]:
    """Parse VASP output files into structures and labels
    By default, the magnetization is read from mag_x from VASP,
    plz modify the code if magnetization is for (y) and (z).

    Args:
        base_dir (str): the directory of the VASP calculation outputs
        check_electronic_convergence (bool): if set to True, this function will raise
            Exception to VASP calculation that did not achieve electronic convergence.
            Default = True
        save_path (str): path to save the parsed VASP labels

    Raises:
        NotADirectoryError: if the base_dir is not a directory

    Returns:
        dict: a dictionary of lists with keys for structure, uncorrected_total_energy,
            energy_per_atom, force, magmom, stress.
    """
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"base_dir={base_dir!r} is not a directory")
    oszicar_path = zpath(f"{base_dir}/OSZICAR")
    vasprun_path = zpath(f"{base_dir}/vasprun.xml")
    outcar_path = zpath(f"{base_dir}/OUTCAR")
    if not os.path.exists(oszicar_path) or not os.path.exists(vasprun_path):
        raise RuntimeError(f"No data parsed from {base_dir}!")
    oszicar = Oszicar(oszicar_path)
    vasprun_orig = Vasprun(
        vasprun_path,
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=False,
    )
    charge, mag_x, mag_y, mag_z, header = [], [], [], [], []
    with zopen(outcar_path, encoding="utf-8") as file:
        all_lines = [line.strip() for line in file.readlines()]
    read_charge = read_mag_x = read_mag_y = read_mag_z = False
    mag_x_all = []
    ion_step_count = 0
    for clean in all_lines:
        if "magnetization (x)" in clean:
            ion_step_count += 1
        if read_charge or read_mag_x or read_mag_y or read_mag_z:
            if clean.startswith("# of ion"):
                header = re.split("\\s{2,}", clean.strip())
                header.pop(0)
            elif re.match("\\s*(\\d+)\\s+(([\\d\\.\\-]+)\\s+)+", clean):
                tokens = [float(token) for token in re.findall("[\\d\\.\\-]+", clean)]
                tokens.pop(0)
                if read_charge:
                    charge.append(dict(zip(header, tokens, strict=True)))
                elif read_mag_x:
                    mag_x.append(dict(zip(header, tokens, strict=True)))
                elif read_mag_y:
                    mag_y.append(dict(zip(header, tokens, strict=True)))
                elif read_mag_z:
                    mag_z.append(dict(zip(header, tokens, strict=True)))
            elif clean.startswith("tot"):
                if ion_step_count == len(mag_x_all) + 1:
                    mag_x_all.append(mag_x)
                read_charge = read_mag_x = read_mag_y = read_mag_z = False
        if clean == "total charge":
            read_charge = True
            read_mag_x = read_mag_y = read_mag_z = False
        elif clean == "magnetization (x)":
            mag_x = []
            read_mag_x = True
            read_charge = read_mag_y = read_mag_z = False
        elif clean == "magnetization (y)":
            mag_y = []
            read_mag_y = True
            read_charge = read_mag_x = read_mag_z = False
        elif clean == "magnetization (z)":
            mag_z = []
            read_mag_z = True
            read_charge = read_mag_x = read_mag_y = False
        elif re.search("electrostatic", clean):
            read_charge = read_mag_x = read_mag_y = read_mag_z = False
    if len(oszicar.ionic_steps) == len(mag_x_all):
        warnings.warn("Unfinished OUTCAR", stacklevel=2)
    elif len(oszicar.ionic_steps) == len(mag_x_all) - 1:
        mag_x_all.pop(-1)
    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])
    dataset = {
        "structure": [],
        "uncorrected_total_energy": [],
        "energy_per_atom": [],
        "force": [],
        "magmom": [],
        "stress": None if "stress" not in vasprun_orig.ionic_steps[0] else [],
    }
    for index, ionic_step in enumerate(vasprun_orig.ionic_steps):
        if (
            check_electronic_convergence
            and len(ionic_step["electronic_steps"]) >= vasprun_orig.parameters["NELM"]
        ):
            continue
        dataset["structure"].append(ionic_step["structure"])
        dataset["uncorrected_total_energy"].append(ionic_step["e_0_energy"])
        dataset["energy_per_atom"].append(ionic_step["e_0_energy"] / n_atoms)
        dataset["force"].append(ionic_step["forces"])
        if mag_x_all != []:
            dataset["magmom"].append([site["tot"] for site in mag_x_all[index]])
        if "stress" in ionic_step:
            dataset["stress"].append(ionic_step["stress"])
    if dataset["uncorrected_total_energy"] == []:
        raise RuntimeError(f"No data parsed from {base_dir}!")
    if save_path is not None:
        save_dict = dataset.copy()
        save_dict["structure"] = [struct.as_dict() for struct in dataset["structure"]]
        write_json(save_dict, save_path)
    return dataset


def solve_charge_by_mag(
    structure: Structure,
    default_ox: (dict[str, float] | None) = None,
    ox_ranges: (dict[str, dict[tuple[float, float], int]] | None) = None,
) -> (Structure | None):
    """Solve oxidation states by magmom.

    Args:
        structure (Structure): pymatgen structure with magmoms in site_properties. Dict
            key must be either magmom or final_magmom.
        default_ox (dict[str, float]): default oxidation state for elements.
            Default = dict(Li=1, O=-2)
        ox_ranges (dict[str, dict[tuple[float, float], int]]): user-defined range to
            convert magmoms into formal valence.
            Example for Mn (Default):
                ("Mn": (
                    (0.5, 1.5): 2,
                    (1.5, 2.5): 3,
                    (2.5, 3.5): 4,
                    (3.5, 4.2): 3,
                    (4.2, 5): 2
                ))

    Returns:
        Structure: pymatgen Structure with oxidation states assigned based on magmoms.
    """
    out_structure = structure.copy()
    out_structure.remove_oxidation_states()
    ox_list = []
    solved_ox = True
    default_ox = default_ox or {"Li": 1, "O": -2}
    ox_ranges = ox_ranges or {
        "Mn": {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}
    }
    magmoms = structure.site_properties.get(
        "final_magmom", structure.site_properties.get("magmom")
    )
    for idx, site in enumerate(out_structure):
        assigned = False
        if site.species_string in ox_ranges:
            for (min_mag, max_mag), mag_ox in ox_ranges[site.species_string].items():
                if min_mag <= magmoms[idx] < max_mag:
                    ox_list.append(mag_ox)
                    assigned = True
                    break
        elif site.species_string in default_ox:
            ox_list.append(default_ox[site.species_string])
            assigned = True
        if not assigned:
            solved_ox = False
    if solved_ox:
        total_charge = sum(ox_list)
        print(f"Solved oxidation state, total_charge={total_charge!r}")
        out_structure.add_oxidation_state_by_site(ox_list)
        return out_structure
    warnings.warn("Failed to solve oxidation state", stacklevel=2)
    return None
