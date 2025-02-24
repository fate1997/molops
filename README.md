# ${\color{#FF9EAA}\rm{Mol}\color{black}\rm{ecule} \space \color{#FF9EAA}\rm{Op}\color{black}\rm{eration}}\color{#FF9EAA}\rm{s}$
This repository contains a lot of useful operation functions to manipulate molecules. Inluding but not limited to:
- Read and write molecular files in various formats.
- Generate and optimize molecular geometries.
- Calculate molecular descriptors.
- Filter and repair molecules.

## Installation
```bash
pip install lapy matplotlib morfeus_ml numpy pandas pint pymatgen scipy seaborn setuptools scikit-image tqdm

conda install -c conda-forge openbabel
conda install -c rdkit rdkit

git clone https://github.com/fate1997/molops.git
cd molops
pip install -e .
```

## Modules

### emol (Enhanced Molecule)
- `EnhancedMol`: Enhanced molecule class. Could be initialized from various formats (including SMILES, XYZ, SDF, PDB, etc.). This class is to extend the functionalities of RDKit Mol class.
```python
from molops.emol import EnhancedMol

emol = EnhancedMol.from_smiles('CCO')
emol = EnhancedMol.from_xyz('XXX.xyz')
emol = EnhancedMol.from_sdf('XXX.sdf')

emol.init_geometry('openbabel')
emol.write_sdf('XXX.sdf')
emol.write_xyz('XXX.xyz')
```

- `EMolContainer`: Enhanced molecules class. Could be initialized from various formats (including CSV, SMILES strings, SDF).
```python
from molops.emol import EMolContainer

emols = EMolContainer.from_smiles_list(['CCO', 'CCN', 'CCF'])
emols = EMolContainer.from_csv('XXX.csv')
emols = EMolContainer.from_sdf('XXX.sdf')

emols.write_sdf('XXX.sdf')
``` 

### descriptors
- `ECFP`: Extended Connectivity Fingerprint.
- `RDKit2D`: RDKit 2D descriptors.
- `RDKit3D`: RDKit 3D descriptors.
```python
from molops.emol import EMolContainer
from molops.descriptors import get_descriptors

emols = EMolContainer.from_smiles(['CCO', 'CCN', 'CCF'])
descriptors = get_descriptors(emols, ['ECFP']) # 'ecfp', 'rdkit2d', 'rdkit3d'
```

### filter & repair
- `DataFilter`: Filter molecules based on given criterions.
```python
from molops.tools import DataFilter
import pandas as pd

df = pd.read_csv('XXX.csv')
filter = DataFilter(
    df = df,
    smiles_col: str = 'smiles',
    target_col: str = 'target',
)
filter.to_cano_smiles()
filter.restrict_target_range(max_val=10, min_val=0)
filter.remove_mixtures()
filter.remove_duplicates()
filter.remove_by_num_elem(min_num_elem=5, max_num_elem=50)
filter.remove_metal_coord()
filter.remove_by_mass(min_mass=50, max_mass=500)

filter.view_distribution()
filter.save('XXX.csv')
```
- `MolRepair`: Repair molecules.
```python
from molops.emol import EMolContainer
from molops.tools import MolRepair

emols = EMolContainer.from_smiles(['CCO', 'CCN', 'CCF'])
repair = MolRepair(
    replace_radical: bool=True,
    canonical_tautomer: bool=True,
    canonical_stereo: bool=False,
    canonical_smiles: bool=True,
    show_tqdm: bool=False,
)
emols = repair.repair(emols)
```

### model
- `XTB`: optimize molecules; generated density file, orbital file, or esp file.
```python
from molops.emol import EnhancedMol
from molops.model import XTBModel

emol = EnhancedMol.from_smiles('CCO')
xtb = XTBModel()
emol = xtb.optimize_emol(emol)

out = xtb.eval(emol, target=['esp', 'density', 'molden'])
```
- `MultiWfn`: calculate esp histogram based on molecular orbitals.
- `PySCF`: obtain esp cube.
