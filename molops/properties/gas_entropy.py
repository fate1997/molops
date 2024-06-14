# Reimplementation of the gas entropy model in the paper:
# https://www.biorxiv.org/content/10.1101/2021.05.26.445640v1.full

import numpy as np
from lapy import TriaMesh
from rdkit import Chem, RDLogger
from skimage import measure

from molops.emol import EnhancedMol

from .register import register_property, require_geometry

RDLogger.DisableLog('rdApp.*')


def gaussian_3D(x: np.array, mol: Chem.Mol, sigma: float=0.1):
    r"""Calculate the 3D Gaussian function for a molecule."""
    coords = mol.GetConformer().GetPositions()
    gaussian_sum = 0
    for atom in mol.GetAtoms():
        vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
        position = coords[atom.GetIdx()]
        gaussian = np.exp(-(np.linalg.norm(position - x, axis=1) - vdw_radius) / sigma)
        gaussian_sum += gaussian
    results = -sigma * np.log(gaussian_sum)
    return results


def iso_surface(mol: Chem.Mol,
                threshold: float,
                spacing: float=0.2,
                padding: float=2,
                sigma: float=0.1):
    r"""Calculate the iso-surface of a molecule."""
    coords = mol.GetConformer().GetPositions()
    min_coords = np.min(coords, axis=0)
    grid_min_coords = min_coords - padding  # in Angstrom
    max_coords = np.max(coords, axis=0)
    grid_max_coords = max_coords + padding  # in Angstrom

    x = np.arange(grid_min_coords[0], grid_max_coords[0] + spacing, spacing)
    y = np.arange(grid_min_coords[1], grid_max_coords[1] + spacing, spacing)
    z = np.arange(grid_min_coords[2], grid_max_coords[2] + spacing, spacing)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    pos = np.stack([xx, yy, zz]).transpose((1, 2, 3, 0)).reshape(-1, 3)
    values = gaussian_3D(pos, mol, sigma).reshape(xx.shape)
    verts, faces, _, _ = measure.marching_cubes(values, threshold, spacing=(spacing, spacing, spacing))
    
    return verts, faces


def get_entropy(values, surface_area_list):
    r"""Calculate the entropy of a series values."""
    prob_density, bins = np.histogram(values, bins=64, density=True)
    prob_per_bin = prob_density * np.diff(bins)
    bins[-1] += 1e-6
    shape_prob_per_face = prob_per_bin[np.digitize(values, bins) - 1]
    shape_entropy = -np.sum(shape_prob_per_face * np.log2(shape_prob_per_face) * np.array(surface_area_list))
    return shape_entropy


def get_shape_entropy(verts, faces):
    r"""Calculate the shape entropy of a molecule."""
    mesh = TriaMesh(verts, faces)
    _, _, k1, k2 = mesh.curvature_tria()
    C_face = 2 / np.pi * np.log((k1**2 + k2**2) / 2)
    S_face = 2 / np.pi * np.arctan((k1 + k2) / (k2 - k1))
    areas_face = mesh.tria_areas()
    
    shape_entropy = get_entropy(C_face, areas_face)
    curvd_entropy = get_entropy(S_face, areas_face)
    return curvd_entropy, shape_entropy


@register_property('gas_entropy')
@require_geometry
def get_gas_entropy(emol: EnhancedMol):
    r"""Estimate the gas phase entropy of a molecule based on its shape."""
    mol = emol.rdmol
    mol = Chem.RemoveHs(mol)
    verts, faces = iso_surface(mol, threshold=0.1)
    curvd_entropy, _ = get_shape_entropy(verts, faces)
    intercept = 70.56867019293395
    scale = 8.70679694
    entropy = intercept + scale * curvd_entropy
    return entropy * 0.001 * 0.93


if __name__ == '__main__':
    gas_entropies = {
        'CCN1C=C[N+](=C1)C': 0.3807,
        '[B-](F)(F)(F)F': 0.2700,
        'C(F)(F)(F)S(=O)(=O)[O-]': 0.3632,
        'CCCCN1C=C[N+](=C1)C': 0.4422,
        'F[P-](F)(F)(F)(F)F': 0.3130
    }
    for smiles, entropy in gas_entropies.items():
        mol = Chem.MolFromSmiles(smiles)
        calculated_entropy = get_gas_entropy(mol)
        print(f'{smiles} gas entropy: {calculated_entropy:.4f} (expected: {entropy:.4f})')