import os
import tempfile
import warnings
from copy import deepcopy
from io import StringIO
from typing import Any, Dict, List, Literal, Tuple

import ase
import ase.io
import numpy as np
import pandas as pd
import py3Dmol
import pyscf
from openbabel import pybel
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D

from molops.emol._parser import parse_cube


class EnhancedMol:
    """EnhancedMol class for handling RDKit Mol objects.
    
    Attributes:
        rdmol (Mol): RDKit Mol object.
        atoms (List[Chem.rdchem.Atom]): List of RDKit Atom objects.
        bonds (List[Chem.rdchem.Bond]): List of RDKit Bond objects.
        geometry (np.ndarray): Molecular geometry.
        smiles (str): SMILES representation.
        num_atoms (int): Number of atoms.
        atom_num (List[int]): Atomic numbers.
        num_conformers (int): Number of conformers.
        num_unpairs (int): Number of unpaired electrons.
        charge (int): Formal charge.
        partial_charges (List[float]): Partial charges.
        components (List[EnhancedMol]): List of components.
        features (Dict[str, Any]): Molecular features or properties.
    """
    def __init__(self, rdmol: Mol, **feature_dict):
        if rdmol is None:
            warnings.warn('Input RDKit Mol is None.')
        self._rdmol = rdmol
        
        self._atoms: List[Chem.rdchem.Atom] = None
        self._bonds: List[Chem.rdchem.Bond] = None
        self._geometry: np.ndarray = None
        self._smiles: str = None
        self._num_atoms: int = None
        self._atom_num: List[int] = None
        self._num_conformers: int = None
        self._num_unpairs: int = None
        self._charge: int = None
        self._components: List['EnhancedMol'] = None
        
        # Molecular features or properties
        if self.rdmol is not None:
            feature_dict.update(self.rdmol.GetPropsAsDict())
        self.features: Dict[str, Any] = feature_dict
        
    @classmethod
    def from_smiles(cls, smiles: str, sanitize: bool=True) -> 'EnhancedMol':
        """Create EnhancedMol object from SMILES string."""
        rdmol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        return cls(rdmol=rdmol)
    
    @classmethod
    def from_sdf(
        cls, 
        path: str, 
        sanitize: bool=True,
        remove_hydrogens: bool=False
    ) -> 'EnhancedMol':
        """Create EnhancedMol object from SDF file."""
        rdmol = Chem.SDMolSupplier(path, sanitize, remove_hydrogens)[0]
        return cls(rdmol=rdmol)
    
    @classmethod
    def from_xyz(cls, path: str, sanitize: bool=True) -> 'EnhancedMol':
        """Create EnhancedMol object from XYZ file."""
        rdmol = Chem.MolFromXYZFile(path)
        if rdmol is None:
            return None
        if sanitize:
            Chem.SanitizeMol(rdmol)
        return cls(rdmol=rdmol)
    
    @classmethod
    def from_cube(cls, path: str) -> 'EnhancedMol':
        """Create EnhancedMol object from cube file."""
        output = parse_cube(path)
        rdmol = Chem.MolFromXYZBlock(output['xyz'])
        emol = cls(rdmol=rdmol)
        emol._partial_charges = output['partial_charges']
        return emol
    
    def __repr__(self) -> str:
        return f'EnhancedMol(num_atoms={self.num_atoms}, ' \
               f'num_conformers={self.num_conformers})' 
    
    def __len__(self) -> int:
        return self.num_atoms
    
    @property
    def is_none(self) -> bool:
        return self.rdmol is None
    
    @property
    def rdmol(self) -> Mol:
        return self._rdmol
    
    @property
    def atoms(self) -> List[Chem.rdchem.Atom]:
        if self._atoms is None:
            self._atoms = [atom for atom in self.rdmol.GetAtoms()]
        return self._atoms
    
    @property
    def bonds(self) -> List[Chem.rdchem.Bond]:
        if self._bonds is None:
            self._bonds = [bond for bond in self.rdmol.GetBonds()]
        return self._bonds
    
    @property
    def geometry(self) -> np.ndarray:
        if self._geometry is None:
            if self.num_conformers > 0:
                self._geometry = self.rdmol.GetConformer().GetPositions()
        return self._geometry
    
    @property
    def smiles(self) -> str:
        if self._smiles is None:
            self._smiles = Chem.MolToSmiles(self.rdmol)
        return self._smiles

    @property
    def atom_num(self) -> List[int]:
        if self._atom_num is None:
            self._atom_num = [atom.GetAtomicNum() for atom in self.atoms]
        return self._atom_num

    @property
    def num_atoms(self) -> int:
        if self._num_atoms is None:
            self._num_atoms = self.rdmol.GetNumAtoms()
        return self._num_atoms
    
    @property
    def num_conformers(self) -> int:
        if self._num_conformers is None:
            self._num_conformers = self.rdmol.GetNumConformers()
        return self._num_conformers

    @property
    def num_unpairs(self) -> int:
        if self._num_unpairs is None:
            unpairs = [atom.GetNumRadicalElectrons() for atom in self.atoms]
            self._num_unpairs = sum(unpairs)
        return self._num_unpairs
    
    @property
    def charge(self) -> int:
        if self._charge is None:
            self._charge = Chem.GetFormalCharge(self.rdmol)
        return self._charge
    
    @property
    def partial_charges(self) -> List[float]:
        atom_dict = self.atoms[0].GetPropsAsDict()
        if 'partial_charges' in atom_dict:
            charges = [float(atom.GetProp('partial_charges')) for atom in self.atoms]
        else:
            print('Partial charges not found. Returning zero charges.')
            charges = [0.0] * self.num_atoms
        return charges
    
    @partial_charges.setter
    def partial_charges(self, charges: List[float]):
        self.add_atom_attr('partial_charges', charges)
    
    @property
    def components(self) -> List['EnhancedMol']:
        if self._components is None:
            components = []
            for component in Chem.GetMolFrags(self.rdmol, asMols=True):
                components.append(EnhancedMol(rdmol=component))
            if len(components) == 1:
                components = None
            self._components = components
        return self._components
    
    @property
    def series(self) -> pd.Series:
        features = self.features.copy()
        feature_names = list(features.keys())
        features['smiles'] = self.smiles
        features = pd.Series(features, index=['smiles'] + feature_names)
        return features
    
    @property
    def atom_attr(self) -> pd.DataFrame:
        attr = {'atom_num': self.atom_num}
        atom_prop_dict = self.atoms[0].GetPropsAsDict()
        for key, value in atom_prop_dict.items():
            if key not in attr and not key.startswith('_'):
                attr[key] = [float(atom.GetProp(key)) for atom in self.atoms]
        return pd.DataFrame(attr)
    
    def add_atom_attr(self, key: str, value: List[float]):
        """Add atom attribute to RDKit Mol object."""
        for atom, v in zip(self.atoms, value):
            atom.SetProp(key, str(v))
    
    def to_pyscf(self, basis: str='6-31g*') -> 'pyscf.gto.Mole':
        """Convert to PySCF Mole object."""
        atom_coords = self.xyz_block.split('\n')[2:]
        atom_coords_str = '; '.join(atom_coords)
        mol = pyscf.gto.Mole()
        mol.atom = atom_coords_str
        mol.basis = basis
        mol.charge = self.charge
        mol.spin = self.num_unpairs
        return mol
    
    def to_ase(self) -> 'ase.Atoms':
        """Convert to ASE Atoms object."""
        atom_coords = self.geometry
        numbers = self.atom_num
        charges = self.partial_charges
        return ase.Atoms(numbers=numbers, positions=atom_coords, charges=charges)
    
    def view2d(
        self,
        figsize: Tuple[int, int]=(300, 300),
        show_index: bool=False,
        **kwargs
    ) -> Image:
        """View 2D representation of molecule."""
        mol = deepcopy(self.rdmol)
        mol.RemoveAllConformers()
        if show_index:
            for i, atom in enumerate(mol.GetAtoms()):
                atom.SetProp('molAtomMapNumber', str(i))
        img = Draw.MolToImage(mol, size=figsize, **kwargs)
        return img
    
    def view3d(
        self,
        figsize: Tuple[int, int]=(800, 400),
    ):
        view = py3Dmol.view(width=figsize[0], height=figsize[1])
        sdf_str = self.sdf_block
        view.addModel(sdf_str, format='sdf')
        view.setStyle({'stick':{}, 'sphere':{'scale': 0.2}},)
        view.zoomTo()
        view.show()
    
    def update_feature(self, key: str, value: any):
        """Update feature."""
        self.features[key] = value
    
    def update_features(self, **kwargs):
        """Update features."""
        self.features.update(kwargs)

    def init_geometry(
        self, 
        method: Literal['rdkit', 'openbabel']
    ) -> 'EnhancedMol':
        """Initialize molecular geometry."""
        if self.geometry is not None:
            warnings.warn('Geometry already initialized. Returning original object.')
            return self
        emol = self.add_hydrogens()
        if method == 'rdkit':
            rdmol = emol.rdmol
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 42
            AllChem.EmbedMolecule(rdmol, ps)
            if rdmol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(rdmol, maxIters=1000)
                emol = EnhancedMol(rdmol=rdmol, **self.features)
        elif method == 'openbabel':
            smiles = emol.smiles
            omol = pybel.readstring("smiles", smiles)
            omol.make3D(steps=1000)
            with tempfile.NamedTemporaryFile(suffix='.sdf') as tf:
                omol.write(format='sdf', filename=tf.name, overwrite=True)
                emol = self.__class__.from_sdf(tf.name)
            emol.features = self.features
        else:
            raise ValueError(f'Unknown method: {method}')
        return emol

    def update_geometry(self, geometry: np.ndarray) -> 'EnhancedMol':
        """Update molecular geometry."""
        rdmol = deepcopy(self.rdmol)
        rdmol.RemoveAllConformers()
        conf = Chem.Conformer(self.num_atoms)
        for i, pos in enumerate(geometry):
            x, y, z = pos.tolist()
            conf.SetAtomPosition(i, Point3D(x, y, z))
        rdmol.AddConformer(conf)
        return self.__class__(rdmol=rdmol, **self.features)
    
    def remove_hydrogens(self) -> 'EnhancedMol':
        rdmol = Chem.RemoveHs(self.rdmol)
        return EnhancedMol(rdmol=rdmol, **self.features)
    
    def add_hydrogens(self) -> 'EnhancedMol':
        rdmol = Chem.AddHs(self.rdmol)
        return EnhancedMol(rdmol=rdmol, **self.features)
    
    def require_conformers(func):
        def wrapper(self, *args, **kwargs):
            if self.num_conformers == 0:
                raise ValueError('No conformers found.')
            return func(self, *args, **kwargs)
        return wrapper
    
    @require_conformers
    def write_sdf(
        self,
        path: str,
        name: str=None,
        feature_names: List[str]=None,
        append: bool=False
    ) -> str:
        mol = deepcopy(self.rdmol)
        if name is not None:
            mol.SetProp('_Name', name)
        if feature_names is not None:
            for feature_name in feature_names:
                mol.SetProp(feature_name, str(self.features[feature_name]))
        # Write to string buffer
        sio = StringIO()
        with Chem.SDWriter(sio) as w:
            w.write(mol)
        
        # Write to file
        with open(path, 'a' if append else 'w') as f:
            f.write(sio.getvalue())
        
        return os.path.abspath(path)
    
    @require_conformers
    def write_xyz(self, path: str) -> str:
        Chem.MolToXYZFile(self.rdmol, path)
        return os.path.abspath(path)
    
    @property
    @require_conformers
    def sdf_block(self) -> str:
        sio = StringIO()
        Chem.SDWriter(sio).write(self.rdmol)
        return sio.getvalue()
    
    @property
    @require_conformers
    def xyz_block(self) -> str:
        return Chem.MolToXYZBlock(self.rdmol)