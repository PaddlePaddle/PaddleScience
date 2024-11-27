from .burgers import load_burgers_1dtime
from .darcy import load_darcy_flow_small
from .darcy import load_darcy_pt
from .navier_stokes import load_navier_stokes_pt
from .pt_dataset import load_pt_traintestsplit
from .spherical_swe import load_spherical_swe

__all__ = [
    "load_burgers_1dtime",
    "load_darcy_flow_small",
    "load_darcy_pt",
    "load_navier_stokes_pt",
    "load_navier_stokes_pt",
    "load_pt_traintestsplit",
    "load_spherical_swe",
]
