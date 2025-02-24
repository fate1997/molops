from molops.descriptors import get_descriptors
from molops.emol import EMolContainer
import pytest
import numpy as np


@pytest.fixture
def emols():
    return EMolContainer.from_smiles_list(['CCO', 'CCN', 'CCF'])


@pytest.mark.parametrize('desc_type', ['ecfp', 'rdkit2d', 'rdkit3d'])
def test_desc(desc_type, emols):
    descriptors, cols = get_descriptors(emols, desc_type, return_columns=True)
    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[0] == 3
    assert len(cols) == descriptors.shape[1]
    assert np.isnan(descriptors).sum() == 0
    assert np.isinf(descriptors).sum() == 0