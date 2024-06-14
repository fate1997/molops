from rdkit.Chem import Descriptors3D

from ._abc import Descriptors
import numpy as np


class RDKit3D(Descriptors):
    r"""Class for RDKit 3D descriptors."""
    
    def calculate_mol(self, mol, config):
        if mol.GetNumConformers() == 0:
            return None
        desc_dict = Descriptors3D.CalcMolDescriptors3D(mol)
        self.set_columns(list(desc_dict.keys()))
        return list(desc_dict.values())