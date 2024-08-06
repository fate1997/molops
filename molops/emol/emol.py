from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from molops.utils.log import without_rdkit_log


@dataclass
class EnhancedMol:
    r"""Class for EnhancedMol object.
    
    Attributes:
        rdmol (Chem.Mol): RDKit Mol object.
        smiles (str, optional): SMILES string. Defaults to None.
    """
    rdmol: Chem.Mol
    smiles: str=None
    
    def __post_init__(self):
        if self.smiles is None:
            self.smiles = Chem.MolToSmiles(self.rdmol)
    
    def __repr__(self):
        return f'EnhancedMol(smiles={self.smiles})'
    
    @property
    def num_conformers(self):
        return self.rdmol.GetNumConformers()
    
    @property
    def components(self):
        r"""Get components of the molecule."""
        if '.' in self.smiles:
            comp_smiles = self.smiles.split('.')
            components = [EnhancedMol.from_source(smiles) for smiles in comp_smiles]
        else:
            components = [self]
        return components  
    
    @property
    def prop(self):
        r"""Get properties of the molecule."""
        prop_dict = self.__dict__.copy()
        prop_dict.pop('rdmol')
        return prop_dict
       
    @classmethod
    def from_source(cls, 
                    source: Union[str, Chem.Mol], 
                    remove_hydrogens: bool=True,
                    standize: bool=False) -> 'EnhancedMol':
        r"""Create EnhancedMol object from source.
        
        Args:
            source (Union[str, Chem.Mol]): Source to create EnhancedMol object.
                The supported sources are SMILES string, SDF file, XYZ file, and PDB file.
            remove_hydrogens (bool, optional): Whether to remove hydrogens. Defaults to True.
            standize (bool, optional): Whether to standize the molecule. Defaults to False.
        
        Returns:
            EnhancedMol: EnhancedMol object.
        """
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
        return cls(rdmol=mol)
    
    def set_rdmol(self, mol: Chem.Mol, update_smiles: bool=True):
        r"""Set RDKit Mol object."""
        self.rdmol = mol
        if update_smiles:
            self.smiles = Chem.MolToSmiles(mol)
    
    def get_img(self,
                figsize: Tuple[int, int]=(300, 300),
                show_index: bool=False,
                **kwargs):
        r"""Get image of the molecule."""
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
        r"""Save the molecule to SDF file.
        
        Args:
            path (str): Path to save the SDF file.
            name (str, optional): Name of the molecule. Defaults to None.
            prop_dict (Dict[str, float], optional): Dictionary of properties. Defaults to None.
            append (bool, optional): Whether to append to the file. Defaults to False.
        """
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
        r"""Standardize the molecule."""
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
    r"""Class for EnhancedMols object."""
    
    emols: List[EnhancedMol]
    # unique_components: List[EnhancedMol]=None
    
    # def __post_init__(self):
    #     if self.unique_components is None:
    #         self.unique_components = list(set(self.emols))
    
    @classmethod
    def from_csv(cls, 
                 csv_path: str,
                 property_cols: List[str]=None,
                 smiles_col: str='smiles',
                 num_samples: int=-1,
                 remove_hydrogens: bool=True,
                 show_tqdm: bool=True,
                 standize: bool=False,
                 random_state: int=42) -> 'EnhancedMols':
        r"""Create EnhancedMols object from CSV file."""
        df = pd.read_csv(csv_path)
        num_samples = len(df) if num_samples == -1 else num_samples
        df = df.sample(num_samples, random_state=random_state)
        source = df[smiles_col].tolist()
        if property_cols is not None:
            properties = {col: df[col].tolist() for col in property_cols}
        else:
            properties = None
        return cls.from_source(source, properties, -1, remove_hydrogens, show_tqdm, standize)
        
    @classmethod
    def from_source(cls, 
                    source: Union[str, List[str], Chem.Mol], 
                    properties: Dict[str, List[float]]=None,
                    num_samples: Union[int, List[int]]=-1,
                    remove_hydrogens: bool=True,
                    show_tqdm: bool=True,
                    standize: bool=False,
                    random_state: int=42) -> 'EnhancedMols':
        r"""Create EnhancedMols object from source.
        
        Args:
            source (Union[str, List[str], Chem.Mol]): Source to create EnhancedMols object.
                The supported sources are SMILES strings, SDF file, and list of RDKit Mol objects.
            properties (Dict[str, List[float]], optional): Dictionary of properties. Defaults to None.
            num_samples (Union[int, List[int]], optional): Number of samples to take. Defaults to -1,
                which means take all samples.
            remove_hydrogens (bool, optional): Whether to remove hydrogens. Defaults to True.
            show_tqdm (bool, optional): Whether to show tqdm progress bar. Defaults to True.
            standize (bool, optional): Whether to standize the molecule. Defaults to False.
            random_state (int, optional): Random state for sampling. Defaults to 42.
        
        Returns:
            EnhancedMols: EnhancedMols object.
        """
        if isinstance(source, str):
            assert source.endswith('.sdf'), 'Only SDF file is supported for now.'
            source = Chem.SDMolSupplier(source, removeHs=remove_hydrogens)
        if num_samples != -1:
            np.random.seed(random_state)
            if isinstance(num_samples, int):
                sampled_ids = np.random.choice(list(range(len(source))), num_samples, replace=False)
            elif isinstance(num_samples, list):
                sampled_ids = num_samples
            source = [source[int(i)] for i in sampled_ids]
            del sampled_ids
        if show_tqdm:
            source = tqdm(source, desc='Loading molecules')
        if properties is not None:
            for value in properties.values():
                assert len(source) == len(value), 'Number of properties should be the same as number of molecules.'
        emols = []
        for i, source in enumerate(source):
            emol = EnhancedMol.from_source(source, remove_hydrogens=remove_hydrogens, standize=standize)
            if properties is not None:
                for k, v in properties.items():
                    emol.set_attr(k, v[i])
            emols.append(emol)
        return cls(emols)

    def __repr__(self):
        return f'EnhancedMols(num_mols={len(self)})'
    
    def __len__(self):
        return len(self.emols)
    
    def __getitem__(self, key) -> EnhancedMol:
        return self.emols[key]
    
    def get_img(self,
                head: int=9,
                figsize: Tuple[int, int]=(300, 300),
                show_index: bool=False,
                legends: List[str]=None,
                **kwargs):
        r"""Get image of the molecules."""
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
    
    def to_sdf(self,
               path: str,
               prop_dict: Dict[str, any]=None):
        r"""Save the molecules to SDF file."""
        if prop_dict is None:
            prop_dict = emol.__dict__
            prop_dict.pop('rdmol')
        for i, emol in enumerate(self.emols):
            emol.to_sdf(path, f'MOL_{i}', prop_dict=prop_dict, append=i!=0)
        return path

    def remove_none(self):
        r"""Remove None values from the list."""
        self.emols = [emol for emol in self.emols if emol is not None]
    
    @property
    def rdmols(self):
        return [emol.rdmol for emol in self.emols]
    
    @property
    def smiles(self):
        return [emol.smiles for emol in self.emols]