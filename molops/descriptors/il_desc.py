import numpy as np
from morfeus import XTB
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
from rdkit.Chem.Descriptors import ExactMolWt, NumRadicalElectrons
from rdkit.Chem.MolStandardize import rdMolStandardize

from ._abc import Descriptors


sp3_smarts = [
    '[!R;CX4H2]', # CH2
    '[!R;CX4H]', # CH
    '[!R;CX4H0]', # C
    '[NX3H1;!R]', # NH
    '[NX3H0;!R]', # N
    '[OX2H0;!R]', # O
    '[#16X2H0;!R]', # S
]

sp2_smarts = [
    '[!R;CX3H1;!$([CX3H1](=O))]', # CH
    '[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]', # C
    '[#7X2H0;!R]', # N
    '[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O' # C=O
]

class ILDesc(Descriptors):
    r"""Class for Ionic Liquid (IL) descriptors."""
    
    def calculate_mol(self, mol, config):
        features = {}
        features.update(self.get_flexibility(mol))
        features.update(self.get_sasa(mol))
        # features.update(self.get_xtb_features(mol))
        features.update(self.get_size_features(mol))
        features.update(self.get_shape_features(mol))
        features.update(self.get_hbond_features(mol))
        features.update(self.get_fluorine_features(mol))
        self.set_columns(list(features.keys()))
        return list(features.values())
    
    @staticmethod
    def get_flexibility(mol, debug=False):
        comps = Chem.GetMolFrags(mol, asMols=True)
        if len(comps) == 1:
            return ILDesc._get_flexibility_from_single_mole(mol, debug)
        elif len(comps) == 2:
            flexibility1 = ILDesc._get_flexibility_from_single_mole(comps[0], debug)
            flexibility2 = ILDesc._get_flexibility_from_single_mole(comps[1], debug)
            return flexibility1 + flexibility2
        else:
            raise ValueError(f'Only support molecules with one or two components.')
    
    @staticmethod
    def _get_flexibility_from_single_mole(mol, debug=False):
        mol = Chem.AddHs(mol)
        # Turn the halogen atoms into H atoms
        halogens = [9, 17, 35, 53]
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in halogens:
                atom.SetAtomicNum(1)        
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        n_sp3 = 0
        n_sp2 = 0
        debug_info = {}
        for smarts in sp3_smarts:
            count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            n_sp3 += count
            if debug and count > 0:
                debug_info[smarts] = count
        for smarts in sp2_smarts:
            count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            n_sp2 += count
            if debug and count > 0:
                debug_info[smarts] = count

        rings = Chem.GetSymmSSSR(mol)
        n_rings = len(rings)
        ring_atoms = [list(ring) for ring in rings]
        # If a ring has common atoms with another ring, adjust the number of rings
        for i in range(len(ring_atoms)):
            for j in range(i+1, len(ring_atoms)):
                if len(set(ring_atoms[i]) & set(ring_atoms[j])) > 0:
                    n_rings -= 1
                    break
        flexibility = n_sp3 + 0.5 * n_sp2 + 0.5 * n_rings - 1
        if flexibility < 0:
            flexibility = 0
        if debug:
            print(debug_info)
            
        # rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        # flexibility_features = {
        #     'flexibility': flexibility,
        #     'rotatable_bonds': rotatable_bonds
        # }
        return flexibility

    @staticmethod
    def get_sasa(mol):
        if mol.GetNumConformers() == 0:
            return None
        radii = rdFreeSASA.classifyAtoms(mol)
        sasa = rdFreeSASA.CalcSASA(mol, radii)
        sasa_features = {
            'sasa': sasa
        }
        return sasa_features

    @staticmethod    
    def get_xtb_features(mol):
        elements = [a.GetAtomicNum() for a in mol.GetAtoms()]
        coordinates = mol.GetConformer(0).GetPositions()
        xtb = XTB(
            elements=elements,
            coordinates=coordinates,
            charge=Chem.GetFormalCharge(mol),
            n_unpaired=NumRadicalElectrons(mol)
        )
        try:
            xtb_features = {
                'dipole': np.linalg.norm(xtb.get_dipole()) * 2.5417,
                'ionisation_potential': xtb.get_ip(corrected=False),
                'homo':xtb.get_homo(),
                'lumo':xtb.get_lumo(),}
        except Exception as e:
            print(e)
            print(mol.GetProp('_Name'))
        return xtb_features

    @staticmethod    
    def get_size_features(mol):
        volume = AllChem.ComputeMolVolume(mol)
        mass = ExactMolWt(mol)
        size_features = {
            'volume': volume,
            'mass': mass
        }
        return size_features
    
    @staticmethod
    def get_symmetry_number(mol, num_confs: int=50):
        comps = Chem.GetMolFrags(mol, asMols=True)
        if len(comps) == 1:
            return ILDesc._get_symmetry_number_from_single_mole(mol, num_confs)
        elif len(comps) == 2:
            return ILDesc._get_symmetry_number_from_binary_mixture(comps[0], comps[1], num_confs)
        else:
            raise ValueError(f'Only support molecules with one or two components.')
    
    @staticmethod
    def _get_symmetry_number_from_binary_mixture(mol1, mol2, num_confs: int=50):
        symmetry_number1 = ILDesc._get_symmetry_number_from_single_mole(mol1, num_confs)
        symmetry_number2 = ILDesc._get_symmetry_number_from_single_mole(mol2, num_confs)
        return np.sqrt(symmetry_number1 * symmetry_number2)

    @staticmethod
    def _get_symmetry_number_from_single_mole(mol, num_confs: int=50):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=100000)
        except:
            return 1
        max_symmetry_number = 1
        elements = [a.GetAtomicNum() for a in mol.GetAtoms()]
        charge = Chem.GetFormalCharge(mol)
        for i in range(mol.GetNumConformers()):
            coordinate = mol.GetConformer(i).GetPositions()
            molecule = Molecule(species=elements, coords=coordinate, charge=charge)
            try:
                symmetry_number = PointGroupAnalyzer(molecule, 
                                                    tolerance=1,
                                                    eigen_tolerance=0.05,
                                                    matrix_tolerance=0.5).get_rotational_symmetry_number()
            except:
                symmetry_number = 1
                # print(f'Warning: Symmetry number calculation failed for {Chem.MolToSmiles(mol)}')
            if symmetry_number > max_symmetry_number:
                max_symmetry_number = symmetry_number
        return max_symmetry_number

    @staticmethod    
    def get_shape_features(mol):
        elements = [a.GetAtomicNum() for a in mol.GetAtoms()]
        coordinates = mol.GetConformer(0).GetPositions()
        molecule = Molecule(
            species=elements,
            coords=coordinates,
            charge=Chem.GetFormalCharge(mol)
        )
        try:
            symmetry_number = PointGroupAnalyzer(molecule, 
                                                 tolerance=1,
                                                 eigen_tolerance=0.05,
                                                 matrix_tolerance=0.5).get_rotational_symmetry_number()
        except:
            symmetry_number = 1
            print(elements, coordinates)
        
        # mol = Chem.RemoveHs(mol)
        # verts, faces = iso_surface(mol, threshold=0.1)
        # curvd_entropy, shape_entropy = get_shape_entropy(verts, faces)
        shape_features = {
            'symmetry_number': symmetry_number,
            # 'curvd_entropy': curvd_entropy,
            # 'shape_entropy': shape_entropy
        }
        return shape_features

    @staticmethod    
    def get_hbond_features(mol):
        num_acceptors = Chem.Lipinski.NumHAcceptors(mol)
        num_donors = Chem.Lipinski.NumHDonors(mol)
        hbond_features = {
            'num_acceptors': num_acceptors,
            'num_donors': num_donors
        }
        return hbond_features

    @staticmethod    
    def get_fluorine_features(mol):
        elements = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        num_fluorine = elements[elements == 9].shape[0]
        fluorine_features = {
            'num_fluorine': num_fluorine
        }
        return fluorine_features