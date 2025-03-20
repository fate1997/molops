import multiprocessing
import os
import subprocess
from typing import List, Literal

from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from molops.emol import EnhancedMol, EMolContainer


class GeometryOptimizer:
    r"""Class for geometry optimization of molecules.
    
    Args:
        method (Literal['UFF', 'MMFF94', 'XTB', 'ETKDG']): Method to use for geometry optimization.
        workdir (str, optional): Working directory for XTB optimization. Defaults to None.
        sdf_path (str, optional): Path to save the SDF file. Defaults to None.
        num_workers (int, optional): Number of workers to use. Defaults to 1.
        remove_hydrogens (bool, optional): Whether to remove hydrogens from the optimized molecule. Defaults to False.
        show_progress (bool, optional): Whether to show tqdm progress bar. Defaults to True.
        **tqdm_kwargs: Additional keyword arguments for tqdm.
    """
    def __init__(
        self, 
        method: Literal['UFF', 'MMFF94', 'XTB', 'ETKDG'], 
        workdir: str=None,
        sdf_path: str=None,
        num_workers: int=1,
        remove_hydrogens: bool=False,
        show_progress: bool=True,
        **tqdm_kwargs
    ):
        self.method = method.upper()
        self.show_progress = show_progress
        self.tqdm_kwargs = tqdm_kwargs
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
        r"""Quickly initialize the geometry using RDKit."""
        mol = emol.rdmol
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        AllChem.EmbedMolecule(mol, ps)
        if mol.GetNumConformers() == 0:
            # print('ETKDG failed for', emol.smiles)
            return None
        if xyz_path is not None:
            Chem.MolToXYZFile(mol, xyz_path)
        return mol
    
    @staticmethod
    def _initialize_by_openbabel(emol: EnhancedMol, xyz_path: str=None):
        r"""Initialize the geometry using OpenBabel."""
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
        r"""Initialize the geometry using the specified method."""
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
        r"""Optimize the geometry using xtb."""
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
                     initial_method: Literal['rdkit', 'openbabel']='rdkit',
                     **kwargs) -> EnhancedMol:
        r"""Optimize the geometry of a molecule."""
        mol = emol.rdmol
        mol = Chem.AddHs(mol)
        emol = EnhancedMol(mol)
        if self.method in ['UFF', 'MMFF94', 'ETKDG']:
            if mol.GetNumConformers() == 0:
                mol = self._initialize_by_rdkit(emol)
                if mol is None:
                    return emol
            if self.method == 'UFF':
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    return emol
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
                print(f'Omol: {[a.GetSymbol() for a in mol.GetAtoms()]}')
                print(f'RDKit: {[a.GetSymbol() for a in optimized_mol.GetAtoms()]}')
                return None
            mol.AddConformer(optimized_mol.GetConformer(0))
        opted_emol = EnhancedMol(mol)
        return opted_emol

    def _optimize_worker(self,
                         emol: EnhancedMol, 
                         cmpd_name: str, 
                         tempdir: str, 
                         level: str, 
                         append_to_sdf: bool):
        r"""Worker function for optimizing a molecule."""
        os.makedirs(tempdir, exist_ok=True)
        optimized_mol = self.optimize_mol(emol, tempdir=tempdir, level=level)
        if optimized_mol is None:
            return None
        if self.sdf_path is not None:
            optimized_mol.write_sdf(self.sdf_path, cmpd_name, append=append_to_sdf)
        return optimized_mol
    
    def _optimize_worker4imap(self, args):
        r"""Wrapper for _optimize_worker for imap."""
        return self._optimize_worker(*args)
    
    def optimize_mols(self, emols: List[EnhancedMol], **kwargs) -> EMolContainer:
        r"""Optimize the geometry of a list of molecules."""
        if self.method == 'XTB':
            max_dir_size = kwargs.get('max_dir_size', 100)
            level = kwargs.get('level', 'normal')
            args = [(emol, f'MOL_{i}', os.path.join(self.workdir, f'temp_{i%max_dir_size}'), level, i!=0)\
                    for i, emol in enumerate(emols)]
            if self.num_workers == 1:
                results = [self._optimize_worker(*arg) for arg in args]
            else:
                with multiprocessing.Pool(self.num_workers) as pool:
                    if self.show_progress:
                        results = list(tqdm(pool.imap(self._optimize_worker4imap, args), 
                                            total=len(args), 
                                            **self.tqdm_kwargs))
                    else:
                        results = pool.starmap(self._optimize_worker, args)
        elif self.method in ['UFF', 'MMFF94', 'ETKDG']:
            emols = [self.optimize_mol(emol) for emol in emols]
            results = EMolContainer(emols)
            if self.sdf_path is not None:
                results.to_sdf(self.sdf_path)
        return results