from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any

from molops.emol import EnhancedMol
from rdkit import Chem
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from dataclasses import dataclass
import multiprocessing


@dataclass
class Descriptors(ABC):
    emols: List[EnhancedMol]
    name: str=None
    columns: List[str]=None
    values: List[List[Union[float, int]]]=None
    config: Dict[str, Any]=None
    show_tqdm: bool=True
    num_workers: int=1
    
    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__ 
        if self.config is None:
            self.config = {}
        if self.values is None:
            self.values = self.calculate_mols(self.emols, self.config)
        if self.columns is None:
            self.columns = [f'{self.name}_{i}' for i in range(len(self.values[0]))]
        elif len(self.columns) != len(self.values[0]):
            raise ValueError(f'Length of columns ({len(self.columns)}) must be equal to length of values ({len(self.values)})')
    
    def __repr__(self):
        return f'{self.name}(num_mols={len(self)}, num_descriptors={len(self.columns)})'
    
    def __len__(self):
        return len(self.values)
    
    def set_columns(self, columns: List[str]):
        self.columns = columns
    
    @property
    def numpy_values(self):
        return np.array(self.values)
    
    @property
    def torch_values(self):
        return torch.tensor(self.numpy_values)
    
    @property
    def df(self):
        return pd.DataFrame(self.numpy_values, columns=self.columns)

    @abstractmethod
    def calculate_mol(self,
                      mol: Chem.Mol, 
                      config: Dict[str, Any]) -> List[Union[float, int]]:
        pass
    
    def calculate_mols(self, 
                       emols: List[EnhancedMol],
                       config: Dict[str, Any]) -> List[List[Union[float, int]]]:
        if self.show_tqdm:
            emols = tqdm(emols, desc=f'{self.name} calculation')
        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                return pool.starmap(self.calculate_mol, [(emol.rdmol, config) for emol in emols])
        else:
            return [self.calculate_mol(emol.rdmol, config) for emol in emols]