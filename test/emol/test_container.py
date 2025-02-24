import pytest

from molops.emol import EMolContainer, EnhancedMol


class TestEmolContainer:
    sdf_path = pytest.EXAMPLE_PATH / 'mols.sdf'
    csv_path = pytest.EXAMPLE_PATH / 'zinc1k.csv'
    
    sdf_emols = None
    csv_emols = None
    
    def test_from_sdf(self):
        emols = EMolContainer.from_sdf(self.sdf_path,
                                       feature_dict={'mw': list(range(5))})
        assert isinstance(emols[0], EnhancedMol)
        assert emols[0].features['mw'] == 0
        assert emols[4].features['mw'] == 4
        assert emols[0].geometry is not None
        self.sdf_emols = emols
        
    def test_from_csv(self):
        emols = EMolContainer.from_csv(self.csv_path,
                                       smiles_col='smiles',
                                       property_cols=['logP', 'qed'],
                                       num_samples=100)
        assert len(emols) == 100
        assert isinstance(emols[0], EnhancedMol)
        assert emols[0].features.get('logP', None) is not None
        assert emols[0].features.get('qed', None) is not None
        assert emols[0].geometry is None
        self.csv_emols = emols