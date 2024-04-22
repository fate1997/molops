from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

from molops.utils.log import without_rdkit_log


@dataclass
class EnhancedMol(Chem.Mol):
    rdmol: Chem.Mol
    
    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(self.rdmol)
    
    @classmethod
    def from_source(cls, 
                    source: str, 
                    remove_hydrogens: bool=True) -> 'EnhancedMol':
        if source.endswith('.sdf'):
            mol = Chem.SDMolSupplier(source)[0]
        elif source.endswith('xyz'):
            mol = Chem.MolFromXYZFile(source)
        elif source.endswith('pdb'):
            mol = Chem.MolFromPDBFile(source)
        else:
            mol = Chem.MolFromSmiles(source)
        
        if mol is None:
            return None
        
        mol = cls.standardize(mol)
        if remove_hydrogens:
            mol = Chem.RemoveHs(mol)
        else:
            mol = Chem.AddHs(mol)
        return cls(rdmol=mol)
    
    def set_rdmol(self, mol: Chem.Mol):
        self.rdmol = mol
    
    def get_img(self,
                figsize: Tuple[int, int]=(300, 300),
                show_index: bool=False,
                **kwargs):
        mol = deepcopy(self.rdmol)
        if show_index:
            for i, atom in enumerate(mol.GetAtoms()):
                atom.SetProp('molAtomMapNumber', str(i))
        img = Draw.MolToImage(mol, size=figsize, **kwargs)
        return img
    
    def to_sdf(self,
               path: str,
               name: str=None,
               prop_dict: Dict[str, float]=None,
               append: bool=False):
        if self.GetNumConformers() == 0:
            raise ValueError('No conformer found. Please generate conformer first.')
        mol = deepcopy(self.rdmol)
        sio = StringIO()
        if name is not None:
            mol.SetProp('_Name', name)
        if prop_dict is not None:
            for k, v in prop_dict.items():
                mol.SetProp(k, str(v))
        with Chem.SDWriter(sio) as w:
            w.write(mol)
        with open(path, 'a' if append else 'w') as f:
            f.write(sio.getvalue())
    
    def add_hydrogens(self, inplace: bool=True):
        mol = Chem.AddHs(self.rdmol)
        if inplace:
            self.set_rdmol(mol)
        return mol
    
    @staticmethod
    def standardize(mol: Chem.Mol, 
                    normalize: bool=True,
                    reionize: bool=True,
                    uncharge: bool=False,
                    stereo: bool=True):
        mol = deepcopy(mol)
        if normalize:
            with without_rdkit_log():
                mol = rdMolStandardize.Normalize(mol)
        if reionize:
            reionizer = rdMolStandardize.Reionizer()
            mol = reionizer.reionize(mol)
        if uncharge:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        if stereo:
            rdmolops.AssignStereochemistry(mol, force=False, cleanIt=True)
        return mol
    
    def set_attr(self, key: str, value):
        setattr(self, key, value)
    
    def get_attr(self, key: str):
        if key in dir(self):
            attr = getattr(self, key)
        else:
            attr = None
        return attr
    

@dataclass
class EnhancedMols:
    emols: List[EnhancedMol]
    
    #!TODO: Implement `from_source`, `__getitem__`, `get_img` method