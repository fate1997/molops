from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator

from ._abc import Descriptors


class RDKit2D(Descriptors):
    
    def calculate_mol(self, mol, config):
        if 'desc_names' not in config:
            desc_names = [x[0] for x in RDKitDescriptors._descList]
            config['desc_names'] = desc_names
        desc_names = config['desc_names']
        if isinstance(desc_names[0], int):
            desc_names = [x[0] for i, x in enumerate(RDKitDescriptors._descList) if i in desc_names]
        calc = MolecularDescriptorCalculator(desc_names)
        descriptors = calc.CalcDescriptors(mol)
        self.set_columns(desc_names)
        return list(descriptors)