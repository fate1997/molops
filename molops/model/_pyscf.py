import os
import shutil
from typing import Tuple

from pyscf import gto, scf
from pyscf.tools import cubegen

from molops.emol import EnhancedMol
from molops.repr import Grid


class PySCFModel:
    def __init__(
        self, 
        basis: str='6-31g*',
        workdir: str='./pyscf_workdir',
        num_workers: int=1
    ):
        self.basis = basis
        self.workdir = workdir
        self.num_workers = num_workers
        
        os.makedirs(self.workdir, exist_ok=True)
    
    def calculate_esp(
        self, 
        emol: EnhancedMol,
        refresh_workdir: bool=False,
        size: Tuple[int, int, int]=(30, 30, 30),
    ) -> Grid:
        if refresh_workdir:
            shutil.rmtree(self.workdir)
            os.makedirs(self.workdir)
        pyscf_mol = emol.to_pyscf(basis=self.basis)
        pyscf_mol.build()
        mf = scf.RHF(pyscf_mol).run()
        path = os.path.join(self.workdir, 'esp.cube')
        cubegen.mep(pyscf_mol, path, mf.make_rdm1(), nx=size[0], ny=size[1], nz=size[2])
        return Grid.from_cube(path)