from rdkit.Chem import Descriptors3D

from ._abc import Descriptors


class RDKit3D(Descriptors):
    
    @staticmethod
    def calculate_mol(mol, config):
        return list(Descriptors3D.CalcMolDescriptors3D(mol).keys())