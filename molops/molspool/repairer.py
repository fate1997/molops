from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from molops.emol import EnhancedMol, EnhancedMols


class MolRepairer:
    r"""Class for repairing molecules.
    
    Args:
        replace_radical (bool, optional): Whether to replace radical atoms with hydrogen atoms. Defaults to True.
        canonical_tautomer (bool, optional): Whether to generate canonical tautomer. Defaults to True.
        canonical_stereo (bool, optional): Whether to generate canonical stereochemistry. Defaults to False.
        canonical_smiles (bool, optional): Whether to generate canonical SMILES. Defaults to True.
        show_tqdm (bool, optional): Whether to show tqdm progress bar. Defaults to False.
    """
    def __init__(self,
                 replace_radical: bool=True,
                 canonical_tautomer: bool=True,
                 canonical_stereo: bool=False,
                 canonical_smiles: bool=True,
                 show_tqdm: bool=False,):
        self.replace_radical = replace_radical
        self.canonical_tautomer = canonical_tautomer
        self.canonical_stereo = canonical_stereo
        self.canonical_smiles = canonical_smiles
        self.show_tqdm = show_tqdm
    
    def repair(self, emols: EnhancedMols):
        r"""Repair the molecules."""
        if self.show_tqdm and len(emols) > 1:
            emols = tqdm(emols)
        repaired_emols = []
        for emol in emols:
            if self.replace_radical:
                emol = self.radical2hydrogen(emol)
            if self.canonical_tautomer:
                ps = rdMolStandardize.CleanupParameters()
                ps.maxTransforms = 5000
                te = rdMolStandardize.TautomerEnumerator(ps)
                rdmol = te.Canonicalize(emol.rdmol)
                emol.set_rdmol(rdmol)
            if self.canonical_stereo:
                Chem.AssignStereochemistry(emol.rdmol, force=True)
            if self.canonical_smiles:
                emol.smiles = Chem.CanonSmiles(emol.smiles)
            repaired_emols.append(emol)
        return EnhancedMols(repaired_emols)
    
    def radical2hydrogen(self, emol: EnhancedMol):
        r"""Replace radical atoms with hydrogen atoms."""
        rdmol = emol.rdmol
        for a in rdmol.GetAtoms():
            if a.GetNumRadicalElectrons() > 0 and a.GetAtomicNum() in [6, 7, 8]:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
                a.UpdatePropertyCache()
        emol.set_rdmol(rdmol)
        return emol