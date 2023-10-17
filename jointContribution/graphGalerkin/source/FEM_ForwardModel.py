import numpy as np
import pdb
import matplotlib.pyplot as plt


from pyCaMOtk.create_mesh_hsphere import mesh_hsphere 
from pyCaMOtk.setup_linelptc_sclr_base_handcode import setup_linelptc_sclr_base_handcode
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_femsp_cg import create_femsp_cg
from pyCaMOtk.solve_fem import solve_fem
from pyCaMOtk.visualize_fem import visualize_fem

def analyticalPossion(xcg,Tc,Tb=0):
    Ue=Tc*(1-xcg[0,:]**2-xcg[1,:]**2)/4+Tb
    return Ue.flatten()

def analyticalConeInterpolation(xcg,Tc,Tb=0):
    Ue=Tc*(1-np.sqrt(xcg[0,:]**2+xcg[1,:]**2))/4+Tb
    return Ue.flatten()