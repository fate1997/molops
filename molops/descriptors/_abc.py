import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from molops.emol import EnhancedMol


@dataclass
class Descriptors(ABC):
    r"""Base class for molecular descriptors.
    
    Attributes:
        emols (List[EnhancedMol]): List of EnhancedMol objects.
        name (str, optional): Name of the descriptor. Defaults to None.
        columns (List[str], optional): List of column names. Defaults to None.
        values (List[List[Union[float, int]]], optional): List of values. Defaults to None.
        config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        show_tqdm (bool, optional): Whether to show tqdm progress bar. Defaults to True.
        num_workers (int, optional): Number of workers to use. Defaults to 1.
    """
    
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
            raise ValueError(f'Length of columns ({len(self.columns)}) must be equal to length of values ({len(self.values[0])})')
    
    def __repr__(self):
        return f'{self.name}(num_mols={len(self)}, num_descriptors={len(self.columns)})'
    
    def __len__(self):
        return len(self.values)
    
    def set_columns(self, columns: List[str]):
        self.columns = columns
    
    @property
    def numpy_values(self):
        r"""Convert the values to numpy array. The None values are converted to np.nan.
        """
        for i, value in enumerate(self.values):
            if value is None:
                self.values[i] = [np.nan] * len(self.columns)
        return np.array(self.values)
    
    @property
    def torch_values(self):
        r"""Convert the values to torch tensor."""
        return torch.from_numpy(self.numpy_values)
    
    @property
    def df(self):
        r"""Convert the values to pandas DataFrame."""
        return pd.DataFrame(self.numpy_values, columns=self.columns)

    @abstractmethod
    def calculate_mol(self,
                      mol: Chem.Mol, 
                      config: Dict[str, Any]) -> List[Union[float, int]]:
        r"""Calculate descriptors for a single RDKit molecule."""
        pass
    
    def calculate_mols(self, 
                       emols: List[EnhancedMol],
                       config: Dict[str, Any]) -> List[List[Union[float, int]]]:
        r"""Calculate descriptors for a list of EnhancedMol objects."""
        if self.show_tqdm:
            emols = tqdm(emols, desc=f'{self.name} calculation')
        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                return pool.starmap(self.calculate_mol, [(emol.rdmol, config) for emol in emols])
        else:
            return [self.calculate_mol(emol.rdmol, config) for emol in emols]