import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from rdkit import Chem

from molops.emol import EnhancedMol


class TestEnhancedMol:
    
    def test_from_rdmol(self):
        rdmol = Chem.MolFromSmiles('CCO')
        feature_dict = {
            'mw': 30.03,
        }
        emol = EnhancedMol(rdmol, **feature_dict)
        assert emol.features['mw'] == 30.03
        assert emol.smiles == 'CCO'
        
    def test_from_smiles(self):
        emol = EnhancedMol.from_smiles('CCO')
        assert isinstance(emol, EnhancedMol)
        assert emol.smiles == 'CCO'
    
    def test_from_sdf(self):
        sdf_path = str(pytest.EXAMPLE_PATH / 'NTF2.sdf')
        emol = EnhancedMol.from_sdf(sdf_path)
        assert isinstance(emol, EnhancedMol)
        assert emol.num_conformers == 1
    
    def test_from_xyz(self):
        xyz_path = str(pytest.EXAMPLE_PATH / 'methanol.xyz')
        emol = EnhancedMol.from_xyz(xyz_path)
        assert isinstance(emol, EnhancedMol)
        assert emol.num_conformers == 1
    
    @pytest.fixture(scope="class")
    def emol(self):
        return EnhancedMol.from_sdf(str(pytest.EXAMPLE_PATH / 'NTF2.sdf'))
    
    def test_num_unpairs(self):
        emol = EnhancedMol.from_smiles('[CH3]')
        assert emol.num_unpairs == 1
    
    def test_components(self):
        emol = EnhancedMol.from_smiles('CC.C')
        assert len(emol.components) == 2
        assert isinstance(emol.components[0], EnhancedMol)
        assert emol.components[0].smiles == 'CC'
        assert emol.components[1].smiles == 'C'

    def test_properties(self, emol: EnhancedMol):
        assert emol.num_conformers == 1
        assert emol.num_atoms == 15
        assert repr(emol) == 'EnhancedMol(num_atoms=15, num_conformers=1)'
        assert len(emol) == 15
        assert len(emol.atoms) == 15
        assert isinstance(emol.atoms[0], Chem.rdchem.Atom)
        assert len(emol.bonds) == 14
        
        assert isinstance(emol.geometry, np.ndarray)
        assert emol.geometry.shape == (15, 3)
        
        assert emol.num_unpairs == 0
        assert emol.charge == -1
        assert emol.components is None
        
        assert emol.features['PUBCHEM_COMPOUND_CID'] == 4176748
        assert isinstance(emol.series, pd.Series)
        assert emol.series['PUBCHEM_COMPOUND_CID'] == 4176748
        
    def test_functions(self, emol: EnhancedMol):
        emol.update_feature('mw', 30.03)
        assert emol.features['mw'] == 30.03
        
        emol.update_features(**{'mw': 30, 'logp': 1.23})
        assert emol.features['mw'] == 30
        assert emol.features['logp'] == 1.23
        
        emol = EnhancedMol.from_smiles('CC')
        emol_hs = emol.add_hydrogens()
        assert emol_hs.num_atoms == 2+6
        emol_nohs = emol_hs.remove_hydrogens()
        assert emol_nohs.num_atoms == 2
        assert emol_nohs.smiles == 'CC'
        
        new_geom = np.array([[0, 0, 0], [1, 0, 0]])
        new_emol = emol.update_geometry(new_geom)
        assert np.allclose(new_emol.geometry, new_geom)
        assert new_emol.features == emol.features
        
    def test_io(self, tmpdir):
        emol = EnhancedMol.from_smiles('CC')
        emol.update_feature('mw', 30)
        emol = emol.update_geometry(np.array([[0, 0, 0], [1, 0, 0]]))
        
        tmp_path = tmpdir.mkdir("emol")
        sdf_path = tmp_path.join("cc.sdf")
        emol.write_sdf(sdf_path, name='test', feature_names=['mw'])
        assert os.path.exists(sdf_path)
        emol_read = EnhancedMol.from_sdf(str(sdf_path))
        assert emol_read.features['mw'] == 30
        assert emol_read.smiles == 'CC'
        
        xyz_path = tmp_path.join("cc.xyz")
        emol.write_xyz(str(xyz_path))
        assert os.path.exists(xyz_path)
        emol_read = EnhancedMol.from_xyz(str(xyz_path))
        assert len(emol_read.atoms) == 2
    
    def test_none(self):
        emol = EnhancedMol(rdmol=None)
        assert emol.is_none
    
    @pytest.mark.parametrize('method', ['rdkit', 'openbabel'])
    def test_init_geometry(self, method):
        emol = EnhancedMol.from_smiles('CC')
        emol.features['mw'] = 30.03
        assert emol.geometry is None
        emol = emol.init_geometry(method)
        assert emol.geometry is not None
        assert emol.geometry.shape == (8, 3)
        assert emol.num_atoms == 8
        assert emol.num_conformers == 1
        assert emol.features['mw'] == 30.03