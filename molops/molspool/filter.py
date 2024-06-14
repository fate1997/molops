from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')


def verbose_decorator(func):
    r"""Decorator for verbose output show the number of rows before and after filtering."""
    def wrapper(self, *args, **kwargs):
        before = len(self.df)
        func(self, *args, **kwargs)
        after = len(self.df)
        print(f'Filter {func.__name__}: {before} -> {after}')
    return wrapper


@dataclass
class DataFilter:
    r"""Class for filtering a dataset of molecules.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing the molecules.
        smiles_col (str): Column name of the SMILES strings.
        target_col (str): Column name of the target values.
    """
    df: pd.DataFrame
    smiles_col: str
    target_col: str
        
    def __post_init__(self):
        smiles_list = self.df[self.smiles_col].tolist()
        self.mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        self.df.insert(0, 'rdmol', self.mols)
    
    @verbose_decorator
    def restrict_target_range(self, max_val: float, min_val: float = 0.0):
        r"""Restrict the target values to a certain range."""
        mask = self.df[self.target_col].between(min_val, max_val, inclusive='both')
        self.df = self.df[mask]
    
    @verbose_decorator
    def to_cano_smiles(self, replace_smiles_col: bool = True):
        r"""Convert the SMILES strings to canonical SMILES."""
        mols = self.df['rdmol']
        mask = np.array([m is not None for m in mols])
        canon_smiles = [Chem.MolToSmiles(m) for m in mols[mask]]
        self.df = self.df[mask]
        if replace_smiles_col:
            self.df[self.smiles_col] = canon_smiles
        else:
            self.df['cano_smiles'] = canon_smiles
            self.smiles_col = 'cano_smiles'
    
    @verbose_decorator
    def remove_mixtures(self, max_components: int = 2):
        r"""Remove mixtures with more than a certain number of components."""
        new_smiles = []
        drop_indices = []
        for i, smiles in enumerate(self.df[self.smiles_col]):
            if '.' in smiles:
                smiles_set = set(smiles.split('.'))
                smiles_set = sorted(smiles_set, key=lambda x: len(x))
                smiles = '.'.join(smiles_set)
                if len(smiles_set) > max_components:
                    drop_indices.append(i)
            new_smiles.append(smiles)
        self.df[self.smiles_col] = new_smiles
        mask = np.array([i not in drop_indices for i in range(len(self.df))])
        self.df = self.df[mask]
    
    @verbose_decorator
    def remove_duplicates(self, max_diff: float = 10.0):
        r"""Remove duplicates and average the target values."""
        mask = self.df[self.smiles_col].duplicated(keep=False)
        duplicates = self.df[mask]
        smiles_groups = duplicates.groupby(self.smiles_col).groups
        drop_indices = []
        for smiles, indices in smiles_groups.items():
            mp = self.df.loc[indices, self.target_col]
            if mp.max() - mp.min() > max_diff:
                drop_indices.extend(indices)
            else:
                self.df.loc[indices[0], self.target_col] = mp.mean()
                drop_indices.extend(indices[1:])
        self.df = self.df.drop(drop_indices)
    
    @verbose_decorator
    def remove_by_num_elem(self, min_num_elem: int=None, max_num_elem: int=None):
        r"""Remove molecules with number of elements outside a certain range."""
        num_elem = [mol.GetNumAtoms() for mol in self.df['rdmol']]
        if min_num_elem is None:
            min_num_elem = 2
        if max_num_elem is None:
            max_num_elem = max(num_elem)
        mask = np.array([min_num_elem <= n <= max_num_elem for n in num_elem])
        self.df = self.df[mask]
    
    @verbose_decorator
    def remove_metal_coord(self):
        r"""Remove molecules with metal coordination."""
        mask = [('->' in smiles) or ('<-' in smiles) for smiles in self.df[self.smiles_col]]
        self.df = self.df[~np.array(mask)]
    
    @verbose_decorator
    def remove_by_mass(self, min_mass: float=None, max_mass: float=None):
        r"""Remove molecules with molecular weight outside a certain range."""
        mol_wts = [Descriptors.MolWt(mol) for mol in self.df['rdmol']]
        if min_mass is None:
            min_mass = 0.0
        if max_mass is None:
            max_mass = max(mol_wts)
        mask = np.array([min_mass <= wt <= max_mass for wt in mol_wts])
        self.df = self.df[mask]
    
    def view_distribution(self, bins: int = 50):
        r"""View the distribution of the target values."""
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(6, 4))
        sns.histplot(self.df[self.target_col].tolist(), bins=bins)
        plt.xlabel(self.target_col)
        plt.ylabel('Count')
        plt.title('Histogram of target values')
        plt.tight_layout()
    
    def save(self, path: str):
        r"""Save the DataFrame to a CSV file."""
        if self.mols is not None:
            self.df = self.df.drop('rdmol', axis=1)
        self.df.to_csv(path, index=False)