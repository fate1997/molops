from rdkit.Chem import Descriptors3D

from ._abc import Descriptors


class RDKit3D(Descriptors):
    
    def calculate_mol(self, mol, config):
        desc_dict = Descriptors3D.CalcMolDescriptors3D(mol)
        self.set_columns(list(desc_dict.keys()))
        return list(desc_dict.values())