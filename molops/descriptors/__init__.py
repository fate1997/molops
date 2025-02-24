from typing import Literal

import numpy as np

from molops.emol import EMolContainer

from .ecfp import ECFP
from .rdkit_2d import RDKit2D
from .rdkit_3d import RDKit3D


def get_descriptors(
    emols: EMolContainer, 
    desc_type: Literal['ecfp', 'rdkit2d', 'rdkit3d'],
    geometry_optimize_method: Literal['rdkit', 'openbabel']='rdkit',
    return_columns: bool=False,
    show_tqdm: bool=True
) -> np.ndarray:
    r"""Get descriptors for a list of EnhancedMol objects.
    
    Args:
        emols: List of EnhancedMol objects.
        desc_type: Type of descriptor to use.
        geometry_optimize_method: Method to use for geometry optimization. 
            Required for 3D descriptors.
        return_columns: If True, return the column names of the descriptors.
        show_tqdm: If True, show a progress bar.
    
    Returns:
        np.ndarray: Numpy array of descriptors.
    """
    desc_type = desc_type.lower()
    if desc_type == 'ecfp':
        desc = ECFP(emols, config={"radius": 4, "nbits": 2048}, show_tqdm=show_tqdm)
    elif desc_type == 'rdkit2d':
        desc = RDKit2D(emols, config={"only_robust": True}, show_tqdm=show_tqdm)
    elif desc_type == 'rdkit3d' or desc_type == 'il_desc':
        if emols[0].num_conformers == 0:
            if geometry_optimize_method is None:
                raise ValueError('RDKit3D descriptor requires geometry optimization method')
            emols = [emol.init_geometry(geometry_optimize_method) for emol in emols]
        if desc_type == 'rdkit3d':
            desc = RDKit3D(emols, show_tqdm=show_tqdm)
    else:
        raise ValueError(f'Invalid descriptor type: {desc_type}')
    if return_columns:
        return desc.numpy_values, desc.columns
    return desc.numpy_values