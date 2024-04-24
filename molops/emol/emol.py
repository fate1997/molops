from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from IPython import display

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

from molops.utils.log import without_rdkit_log


@dataclass
class EnhancedMol:
    rdmol: Chem.Mol
    smiles: str=None
    
    def __post_init__(self):
        if self.smiles is None:
            self.smiles = Chem.MolToSmiles(self.rdmol)
    
    def __repr__(self):
        img = self.get_img()
        display.display(img)
        return self.smiles
    
    @classmethod
    def from_source(cls, 
                    source: Union[str, Chem.Mol], 
                    remove_hydrogens: bool=True,
                    standize: bool=False) -> 'EnhancedMol':
        if isinstance(source, str):
            if source.endswith('.sdf'):
                mol = Chem.SDMolSupplier(source)[0]
            elif source.endswith('xyz'):
                mol = Chem.MolFromXYZFile(source)
            elif source.endswith('pdb'):
                mol = Chem.MolFromPDBFile(source)
            else:
                mol = Chem.MolFromSmiles(source)
                return cls(rdmol=mol, smiles=source)
        else:
            mol = source
        
        if mol is None:
            return None
        if standize:
            with without_rdkit_log():
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
        mol.RemoveAllConformers()
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
        mol = deepcopy(self.rdmol)
        if mol.GetNumConformers() == 0:
            raise ValueError('No conformer found. Please generate conformer first.')
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
    
    @classmethod
    def from_source(cls, 
                    source: Union[str, List[str]], 
                    remove_hydrogens: bool=True,
                    show_tqdm: bool=True,
                    standize: bool=False) -> 'EnhancedMols':
        if isinstance(source, str):
            assert source.endswith('.sdf'), 'Only SDF file is supported for now.'
            source = Chem.SDMolSupplier(source)
        if show_tqdm:
            source = tqdm(source, desc='Loading molecules')
        emols = [EnhancedMol.from_source(s, remove_hydrogens, standize) for s in source]
        return cls(emols)

    def __repr__(self):
        return f'EnhancedMols(num_mols={len(self)})'
    
    def __len__(self):
        return len(self.emols)
    
    def __getitem__(self, key):
        return self.emols[key]
    
    def get_img(self,
                head: int=9,
                figsize: Tuple[int, int]=(300, 300),
                show_index: bool=False,
                legends: List[str]=None,
                **kwargs):
        rdmols = [emol.rdmol for emol in self.emols[:head]]
        if show_index:
            for rdmol in rdmols:
                for j, atom in enumerate(rdmol.GetAtoms()):
                    atom.SetProp('molAtomMapNumber', str(j))
        if legends is None:
            legends = [f'ID: {i}' for i in range(len(rdmols))]
        img = Draw.MolsToGridImage(rdmols,
                                   subImgSize=figsize,
                                   legends=legends,
                                   **kwargs)
        return img