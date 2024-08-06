# ${\color{#FF9EAA}\rm{Mol}\color{black}\rm{ecule} \space \color{#FF9EAA}\rm{Op}\color{black}\rm{eration}}\color{#FF9EAA}\rm{s}$
This repository contains a lot of useful operation functions to manipulate molecules. Inluding but not limited to:
- [x] Read and write molecular files in various formats.
- [x] Calculate molecular properties.
- [x] Generate and optimize molecular geometries.
- [x] Calculate molecular descriptors.
- [x] Filter and repair molecules.

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
### descriptors
- [x] `ECFP`: Extended Connectivity Fingerprint.
- [x] `RDKit2D`: RDKit 2D descriptors.
- [x] `RDKit3D`: RDKit 3D descriptors.

### emol (Enhanced Molecule)
- [x] `EnhancedMol`: Enhanced molecule class. Could be initialized from various formats (including SMILES, XYZ, SDF, PDB, etc.). This class is to extend the functionalities of RDKit Mol class.
- [x] `EnhancedMols`: Enhanced molecules class. Could be initialized from various formats (including CSV, SMILES strings, SDF).

### filter & repair
- [x] `DataFilter`: Filter molecules based on given criterions.
- [x] `MolRepair`: Repair molecules.

### geometry optimization (`GeometryOptimizer`)
- [x] ETKDG and OpenBabel geometry initialization.
- [x] UFF, MMFF94 force field optimization.
- [x] XTB optimization (parallelized).
