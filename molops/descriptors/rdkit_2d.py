from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator

from ._abc import Descriptors


class RDKit2D(Descriptors):
    r"""Class for RDKit 2D descriptors."""
    
    def calculate_mol(self, mol, config):
        desc_names = [x[0] for x in RDKitDescriptors._descList]
        if 'desc_names' in config:
            desc_names = config['desc_names']
        
        if 'only_robust' in config and 'desc_names' not in config:
            if config['only_robust']:
                unrobust_indices = [10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25]
                desc_names = [name for i, name in enumerate(desc_names) \
                                        if i not in unrobust_indices]
        if isinstance(desc_names[0], int):
            desc_names = [x[0] for i, x in enumerate(RDKitDescriptors._descList) if i in desc_names]
        calc = MolecularDescriptorCalculator(desc_names)
        descriptors = calc.CalcDescriptors(mol)
        self.set_columns(desc_names)
        return list(descriptors)