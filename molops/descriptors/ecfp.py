from rdkit.Chem import AllChem

from ._abc import Descriptors


class ECFP(Descriptors):
    
    def calculate_mol(self, mol, config):
        radius = config['radius']
        nBits = config['nbits']
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))