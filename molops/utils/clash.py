import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist


def get_vdw_radii(atom_num: int) -> np.ndarray:
    r"""Get the vdw radius for a given atom number.
    """
    
    # VDW radii data from ase package
    radius_list = np.array([np.nan, 1.2 , 1.4 , 1.82, 1.53, 1.92, 1.7 , 1.55, 1.52, 1.47, 1.54,
       2.27, 1.73, 1.84, 2.1 , 1.8 , 1.8 , 1.75, 1.88, 2.75, 2.31,  np.nan,
        np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan, 1.63, 1.4 , 1.39, 1.87, 2.11,
       1.85, 1.9 , 1.85, 2.02, 3.03, 2.49,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,
        np.nan,  np.nan, 1.63, 1.72, 1.58, 1.93, 2.17, 2.06, 2.06, 1.98, 2.16,
       3.43, 2.49,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  
       np.nan, np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  
       np.nan,  np.nan,  np.nan, 1.75, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.2 , 3.48,
       2.83,  np.nan,  np.nan,  np.nan, 1.86,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  
       np.nan, np.nan,  np.nan,  np.nan,  np.nan,  np.nan])
    
    radius = radius_list[atom_num]

    return radius

def is_steric_clash(rdmol: Chem.Mol, scale=1.2):
    r"""Check if there is steric clash in the molecule.
    """
    coordinates = rdmol.GetConformer().GetPositions()
    dist_matrix = cdist(coordinates, coordinates)
    elements = np.array([a.GetAtomicNum() for a in rdmol.GetAtoms()])
    adj_matrix = rdmol.GetAdjacencyMatrix()
    num_atoms = rdmol.GetNumAtoms()
    mask = np.array([row for row in adj_matrix]) + np.eye(num_atoms)
    mask = ~mask.astype(bool)
    
    # Get the average vdw radii of the atoms in the base and candidate
    vdw_radii = get_vdw_radii(elements)
    average_vdw_radii = (vdw_radii[:, np.newaxis] + vdw_radii) / 2
    
    # Get the threshold for steric clash
    threshold = (average_vdw_radii * scale)[np.newaxis, :, :] * mask
    
    return (dist_matrix < threshold).any(axis=-1).any(axis=-1)