import pickle
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

from molops.emol import EnhancedMol


class EMolContainer:
    """A container for EnhancedMol objects.
    
    Attributes:
        emols (List[EnhancedMol]): A list of EnhancedMol objects.
        df (pd.DataFrame): A DataFrame representation of the molecules.
        features (Dict[str, List[Any]]): Molecule features, the keys are the 
            feature names and the values are lists of feature values.
    """
    def __init__(self, emols: List[EnhancedMol]):
        self.emols = emols
        
        self._df = None
        self._features = None
        
    @classmethod
    def from_rdmols(
        cls,
        rdmols: List[Chem.rdchem.Mol],
        num_samples: int=-1,
        feature_dict: Dict[str, List[Any]]=None
    ) -> 'EMolContainer':
        """Create an EMolContainer from a list of RDKit molecules."""
        if num_samples != -1:
            selected_indices = np.random.choice(len(rdmols), num_samples, replace=False)
        else:
            selected_indices = list(range(len(rdmols)))
        emols = []
        for i in selected_indices:
            rdmol = rdmols[i]
            if feature_dict is not None:
                feature = {k: v[i] for k, v in feature_dict.items()}
            else:
                feature = {}
            emol = EnhancedMol(rdmol, **feature)
            emols.append(emol)
        return cls(emols)
    
    @classmethod
    def from_sdf(
        cls,
        path: str,
        sanitize: bool=True,
        remove_hydrogens: bool=False,
        num_samples: int=-1,
        feature_dict: Dict[str, List[Any]]=None
    ) -> 'EMolContainer':
        """Create an EMolContainer from an SDF file."""
        rdmols = Chem.SDMolSupplier(path, sanitize, remove_hydrogens)
        return cls.from_rdmols(rdmols, num_samples, feature_dict)
    
    @classmethod
    def from_smiles_list(
        cls,
        smiles_list: List[str],
        sanitize: bool=True,
        num_samples: int=-1,
        feature_dict: Dict[str, List[Any]]=None
    ) -> 'EMolContainer':
        """Create an EMolContainer from a list of SMILES strings."""
        rdmols = []
        for i in range(len(smiles_list)):
            rdmol = Chem.MolFromSmiles(smiles_list[i], sanitize=sanitize)
            if rdmol is not None:
                rdmols.append(rdmol)
        return cls.from_rdmols(rdmols, num_samples, feature_dict)
    
    @classmethod
    def from_csv(
        cls, 
        csv_path: str,
        smiles_col: str='smiles',
        property_cols: List[str]=None,
        sanitize: bool=True,
        num_samples: int=-1
    ) -> 'EMolContainer':
        """Create an EMolContainer from a CSV file."""
        df = pd.read_csv(csv_path)
        smiles_list = df[smiles_col].tolist()
        if property_cols is not None:
            feature_dict = {col: df[col].tolist() for col in property_cols}
        else:
            feature_dict = None
        return cls.from_smiles_list(smiles_list, sanitize, num_samples, feature_dict)
    
    @classmethod
    def from_txt(
        cls,
        txt_path: str,
        header: bool=False,
        sanitize: bool=True,
        num_samples: int=-1
    ) -> 'EMolContainer':
        """Create an EMolContainer from a text file."""
        smiles_list = []
        with open(txt_path, 'r') as f:
            if header:
                next(f)
            for line in f:
                smiles_list.append(line.strip())
        return cls.from_smiles_list(smiles_list, sanitize, num_samples)

    @classmethod
    def from_pickle(cls, path: str) -> 'EMolContainer':
        """Load an EMolContainer from a pickle file."""
        import pickle
        with open(path, 'rb') as f:
            emols = pickle.load(f)
        return emols
    
    @property
    def df(self) -> pd.DataFrame:
        """Show the DataFrame representation of the molecules."""
        if self._df is None:
            series_list = [emol.series for emol in self.emols]
            self._df = pd.DataFrame(series_list)
        return self._df
    
    @property
    def features(self) -> Dict[str, List[Any]]:
        if self._features is None:
            features_ = {}
            for emol in self.emols:
                for k, v in emol.features.items():
                    if k not in features_:
                        features_[k] = []
                    features_[k].append(v)
            self._features = features_
        return self._features
    
    def __repr__(self):
        return f'EMolContainer(num_mols={len(self)})'
    
    def __len__(self):
        return len(self.emols)
    
    def __getitem__(self, key) -> EnhancedMol:
        return self.emols[key]
    
    def get_img(
        self,
        head: int=9,
        figsize: Tuple[int, int]=(300, 300),
        show_index: bool=False,
        legends: List[str]=None,
        **kwargs
    ) -> Image.Image:
        rdmols = [emol.rdmol for emol in self.emols[:head]]
        for rdmol in rdmols:
            if show_index:
                for j, atom in enumerate(rdmol.GetAtoms()):
                    atom.SetProp('molAtomMapNumber', str(j))
            rdmol.RemoveAllConformers()
            
        if legends is None:
            legends = [f'ID: {i}' for i in range(len(rdmols))]
        img = Draw.MolsToGridImage(rdmols,
                                   subImgSize=figsize,
                                   legends=legends,
                                   **kwargs)
        return img
    
    def write_sdf(self, path: str, feature_names: List[str]=None) -> str:
        r"""Save the molecules to SDF file."""
        for i, emol in enumerate(self.emols):
            if emol is None or emol.rdmol is None:
                continue
            emol.write_sdf(path, f'MOL_{i}', feature_names, append=i!=0)
        return path
    
    @property
    def rdmols(self) -> List[Chem.rdchem.Mol]:
        return [emol.rdmol for emol in self.emols]
    
    @property
    def smiles(self) -> List[str]:
        return [emol.smiles for emol in self.emols]
    
    def apply(
        self, 
        func: Callable[[EnhancedMol], Any], 
        show_progress: bool=False,
        remove_none: bool=True,
        **kwargs
    ) -> List[Any]:
        emols = self.emols
        if show_progress:
            emols = tqdm(emols, desc='Applying function')
        results = []
        for emol in emols:
            try:
                result = func(emol, **kwargs)
            except Exception as e:
                print(f'Error applying function to molecule {emol.smiles}: {e}')
                result = None
            if result is not None or not remove_none:
                results.append(result)
        return results
    
    def update(self, func: Callable[[EnhancedMol], EnhancedMol]) -> 'EMolContainer':
        new_emols = []
        for emol in self.emols:
            new_emols.append(func(emol))
        return self.__class__(new_emols)
    
    def add_hydrogens(self) -> 'EMolContainer':
        return self.update(lambda emol: emol.add_hydrogens())
    
    def remove_hydrogens(self) -> 'EMolContainer':
        return self.update(lambda emol: emol.remove_hydrogens())
    
    def write_pickle(self, path: str) -> str:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return path