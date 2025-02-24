import tempfile

from molops import EnhancedMol
from molops.model import PySCFModel, XTBModel
from molops.repr import Grid


def test_calculate_esp():
    emol = EnhancedMol.from_smiles('CCO')
    with tempfile.TemporaryDirectory() as workdir:
        xtb = XTBModel(workdir=workdir)
        emol = xtb.optimize_emol(emol)
        pyscf_model = PySCFModel(workdir=workdir)
        grid = pyscf_model.calculate_esp(emol)
        assert isinstance(grid, Grid)