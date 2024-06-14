from typing import Literal

import numpy as np

from molops.emol import EnhancedMols, GeometryOptimizer

from .ecfp import ECFP
from .il_desc import ILDesc
from .rdkit_2d import RDKit2D
from .rdkit_3d import RDKit3D


def get_descriptors(emols: EnhancedMols, 
                    desc_type: Literal['ecfp', 'rdkit2d', 'rdkit3d', 'il_desc'],
                    geometry_optimize_method: Literal['UFF', 'MMFF94', 'XTB', 'ETKDG']='ETKDG',
                    return_columns: bool=False) -> np.ndarray:
    r"""Get descriptors for a list of EnhancedMol objects.
    
    Args:
        emols: List of EnhancedMol objects.
        desc_type: Type of descriptor to use.
        geometry_optimize_method: Method to use for geometry optimization. 
            Required for 3D descriptors.
        return_columns: If True, return the column names of the descriptors.
    
    Returns:
        np.ndarray: Numpy array of descriptors.
    """
    if desc_type == 'ecfp':
        desc = ECFP(emols, config={"radius": 4, "nbits": 2048})
    elif desc_type == 'rdkit2d':
        desc = RDKit2D(emols, config={"only_robust": True})
    elif desc_type == 'rdkit3d' or desc_type == 'il_desc':
        if emols[0].rdmol.GetNumConformers() == 0:
            if geometry_optimize_method is None:
                raise ValueError('RDKit3D descriptor requires geometry optimization method')
            geom_opt = GeometryOptimizer(method=geometry_optimize_method)
            emols = geom_opt.optimize_mols(emols)
        if desc_type == 'rdkit3d':
            desc = RDKit3D(emols)
        elif desc_type == 'il_desc':
            desc = ILDesc(emols)
    else:
        raise ValueError(f'Invalid descriptor type: {desc_type}')
    if return_columns:
        return desc.numpy_values, desc.columns
    return desc.numpy_values