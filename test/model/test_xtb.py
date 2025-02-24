import pytest
import numpy as np

from molops.model import XTBModel
from molops import EnhancedMol
from molops.repr import PointCloud, Grid


@pytest.fixture
def xtb_model():
    return XTBModel()


def test_optimize(xtb_model: XTBModel):
    emol = EnhancedMol.from_smiles('CCO')
    optimized_emol = xtb_model.optimize_emol(emol)
    assert optimized_emol.num_conformers == 1
    assert np.isnan(optimized_emol.geometry).sum() == 0
    assert optimized_emol.geometry.shape == (optimized_emol.num_atoms, 3)
    assert optimized_emol.partial_charges is not None
    assert len(optimized_emol.partial_charges) == optimized_emol.num_atoms


def test_eval(xtb_model: XTBModel):
    emol = EnhancedMol.from_smiles('CCO')
    emol = xtb_model.optimize_emol(emol)
    out = xtb_model.eval(emol, target=['esp', 'density', 'molden'])
    assert 'esp' in out
    assert 'density' in out
    assert 'molden' in out
    assert isinstance(out['esp'], PointCloud)
    assert isinstance(out['density'], Grid)
    assert isinstance(out['molden'], str)