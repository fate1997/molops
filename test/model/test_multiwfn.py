from molops.model import XTBModel, MultiWfnModel
from molops import EnhancedMol
import pandas as pd

import tempfile
import os


def test_get_esp_hist():
    emol = EnhancedMol.from_smiles('CCO')
    xtb_model = XTBModel()
    emol = xtb_model.optimize_emol(emol)
    out = xtb_model.eval(emol, target=['molden'])
    
    # Context manager for temporary directory
    with tempfile.TemporaryDirectory() as workdir:
        multiwfn_model = MultiWfnModel(work_dir=workdir)
        molden_path = os.path.join(workdir, 'molden.input')
        with open(molden_path, 'w') as f:
            f.write(out['molden'])
        esp_hist = multiwfn_model.get_esp_hist(molden_path)
        assert isinstance(esp_hist, pd.DataFrame)