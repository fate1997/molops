from rdkit import Chem
from molops.emol import EnhancedMol
from .register import register_property


# From https://journals.iucr.org/b/issues/2002/03/02/bk0101/index.html
ATOM_VOLUME_CONTRIBUTIONS = {
    1: 5.08,
    3: 22.6,
    4: 36,
    5: 13.24,
    6: 13.87,
    7: 11.8,
    8: 11.39,
    9: 11.17,
    11: 26,
    12: 36,
    13: 39.6,
    14: 37.3,
    15: 29.5,
    16: 25.2,
    17: 25.8, 
    19: 36,
    20: 45,
    21: 42,
    22: 27.3,
    23: 24,
    24: 28.1,
    25: 31.9,
    26: 30.4,
    27: 29.4,
    28: 26, 
    29: 26.9,
    33: 36.4,
    34: 30.3,
    35: 32.7,
    40: 27,
    49: 55,
    50: 52.8,
    51: 48,
    52: 46.7,
    53: 46.2,
    80: 38.0,
}


@register_property('crystal_volume')
def get_crystal_volume(emol: EnhancedMol):
    r"""Estimate the crystal volume of a molecule based on atomic contributions."""
    mol = emol.rdmol
    volume = 0
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        volume += ATOM_VOLUME_CONTRIBUTIONS[atomic_num] * 1e-3 # Convert A^3 to nm^3
    return volume


if __name__ == '__main__':
    smiles_volume = {
        'CCN1C=C[N+](=C1)C': 0.156,
        '[B-](F)(F)(F)F': 0.073,
        'C(F)(F)(F)S(=O)(=O)[O-]': 0.131,
        'C(F)(F)(F)S(=NS(=O)(=O)C(F)(F)F)(=O)[O-]': 0.232,
        'CCCCN1C=C[N+](=C1)C': 0.211
    }
    for smiles, volume in smiles_volume.items():
        mol = Chem.MolFromSmiles(smiles)
        print(f'{smiles} volume: {get_crystal_volume(mol):.3f} (expected: {volume:.3f})')