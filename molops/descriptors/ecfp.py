from rdkit.Chem import AllChem

from ._abc import Descriptors


class ECFP(Descriptors):
    
    @staticmethod
    def calculate_mol(mol, config):
        radius = config['radius']
        nBits = config['nbits']
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))