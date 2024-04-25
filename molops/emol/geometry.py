import logging
import multiprocessing
import os
import subprocess
from functools import partial
from io import StringIO
from typing import Literal, List, Union
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

from molops.emol import EnhancedMol, EnhancedMols


class GeometryOptimizer:
    def __init__(self, 
                 method: Literal['UFF', 'MMFF94', 'XTB', 'ETKDG'], 
                 workdir: str=None,
                 sdf_path: str=None,
                 num_workers: int=1,
                 remove_hydrogens: bool=False):
        self.method = method.upper()
        if method == 'XTB':
            workdir = './tmp' if workdir is None else workdir
            os.makedirs(workdir, exist_ok=True)
        self.workdir = os.path.abspath(workdir) if workdir is not None else None
        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        self.num_workers = num_workers
        self.remove_hydrogens = remove_hydrogens
        self.sdf_path = os.path.abspath(sdf_path) if sdf_path is not None else None
    
    @staticmethod
    def _initialize_by_rdkit(emol: EnhancedMol, xyz_path: str=None) -> Chem.Mol:
        mol = emol.rdmol
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        AllChem.EmbedMolecule(mol, ps)
        if mol.GetNumConformers() == 0:
            print('ETKDG failed for', emol.smiles)
            return
        if xyz_path is not None:
            Chem.MolToXYZFile(mol, xyz_path)
        return mol
    
    @staticmethod
    def _initialize_by_openbabel(emol: EnhancedMol, xyz_path: str=None):
        smiles = emol.smiles
        omol = pybel.readstring("smiles", smiles)
        omol.make3D(steps=5000)
        if xyz_path is not None:
            omol.write("xyz", xyz_path, overwrite=True)
        return omol
    
    def initialize_geometry(self, 
                            emol: EnhancedMol, 
                            method: Literal['rdkit', 'openbabel'],
                            xyz_path: str=None) -> str:
        if method == 'rdkit':
            self._initialize_by_rdkit(emol, xyz_path)
        elif method == 'openbabel':
            self._initialize_by_openbabel(emol, xyz_path)
        else:
            raise ValueError(f'Unknown method: {method}')
        return os.path.abspath(xyz_path)
    
    def _optimize_by_xtb(self, 
                         xyz_path: str, 
                         tempdir: str,
                         level: str = 'normal', 
                         charge: int=0):
        original_dir = os.path.abspath(os.getcwd())
        os.makedirs(tempdir, exist_ok=True)
        os.chdir(tempdir)
        results = subprocess.run(["xtb", xyz_path, "--silent", "--opt", level, f"--charge", f"{charge}"],
                                check=True,
                                capture_output=True,
                                text=True,)
        with open('xtb.stdout', "w") as f:
            f.write(results.stdout)
        with open('xtb.stderr', "w") as f:
            f.write(results.stderr)
        optimized_path = os.path.abspath('xtbopt.xyz')
        os.chdir(original_dir)
        return optimized_path
    
    def optimize_mol(self, 
                     emol: EnhancedMol, 
                     initial_method: Literal['rdkit', 'openbabel']='openbabel',
                     **kwargs) -> EnhancedMol:
        mol = emol.rdmol
        mol = Chem.AddHs(mol)
        if self.method in ['UFF', 'MMFF94', 'ETKDG']:
            if mol.GetNumConformers() == 0:
                mol = self._initialize_by_rdkit(emol)
            if self.method == 'UFF':
                AllChem.UFFOptimizeMolecule(mol)
            if self.method == 'MMFF94':
                AllChem.MMFFOptimizeMolecule(mol)
        elif self.method == 'XTB':
            tempdir = kwargs.get('tempdir', self.workdir)
            level = kwargs.get('level', 'normal')
            xyz_path = os.path.join(tempdir, 'original.xyz')
            xyz_path = os.path.abspath(xyz_path)
            xyz_path = self.initialize_geometry(emol, initial_method, xyz_path)
            try:
                optimized_path = self._optimize_by_xtb(xyz_path, 
                                                    tempdir, 
                                                    level=level,
                                                    charge=Chem.GetFormalCharge(mol))
            except subprocess.CalledProcessError as e:
                print(f'Error in xtb optimization: {e}')
                return None
            optimized_mol = Chem.MolFromXYZFile(optimized_path)
            if [a.GetSymbol() for a in mol.GetAtoms()] != \
                [a.GetSymbol() for a in optimized_mol.GetAtoms()]:
                print(f'Omol and RDKit molecule atoms do not match')
                return None
            mol.AddConformer(optimized_mol.GetConformer(0))
        emol = EnhancedMol.from_source(mol, remove_hydrogens=self.remove_hydrogens)
        return emol

    def _optimize_worker(self,
                         emol: EnhancedMol, 
                         cmpd_name: str, 
                         tempdir: str, 
                         level: str, 
                         append_to_sdf: bool):
        os.makedirs(tempdir, exist_ok=True)
        optimized_mol = self.optimize_mol(emol, tempdir=tempdir, level=level)
        if optimized_mol is None:
            return None
        emol = EnhancedMol.from_source(optimized_mol, remove_hydrogens=False)
        emol.to_sdf(self.sdf_path, cmpd_name, append=append_to_sdf)
        return emol
    
    def optimize_mols(self, emols: List[EnhancedMol], **kwargs) -> EnhancedMols:
        if self.method == 'XTB':
            max_dir_size = kwargs.get('max_dir_size', 100)
            level = kwargs.get('level', 'normal')
            args = [(emol, f'MOL_{i}', os.path.join(self.workdir, f'temp_{i%max_dir_size}'), level, i!=0)\
                    for i, emol in enumerate(emols)]
            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.starmap(self._optimize_worker, args)
        elif self.method in ['UFF', 'MMFF94', 'ETKDG']:
            emols = [self.optimize_mol(emol) for emol in emols]
            results = EnhancedMols(emols)
        return results