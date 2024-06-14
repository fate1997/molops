from rdkit.Chem import AllChem

from ._abc import Descriptors


class ECFP(Descriptors):
    r"""Class for Extended Connectivity Fingerprint (ECFP) descriptors."""
    def calculate_mol(self, mol, config):
        radius = config['radius']
        nBits = config['nbits']
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))