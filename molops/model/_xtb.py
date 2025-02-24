import os
import pathlib
import shutil
import subprocess
from typing import Any, Dict, List, Literal

import numpy as np

from molops.emol import EnhancedMol
from molops.repr import Grid, PointCloud
from molops.utils.facility import run_in

XTB_INP = str(pathlib.Path(__file__).parent / '_xtb.inp')


class XTBModel:
    def __init__(
        self,
        xtb_path: str='xtb',
        workdir: str='./xtb_workdir',
        num_workers: int=1,
        verbose: bool=True
    ):
        self.xtb_path = xtb_path
        self.workdir = workdir
        if num_workers == -1:
            num_workers = os.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(num_workers)
        os.environ['MKL_NUM_THREADS'] = str(num_workers)
        # Clean up workdir
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir)
        self.verbose = verbose
        self._stdouts = {}
        self._stderrs = {}
       
    def optimize_emol(
        self,
        emol: EnhancedMol,
        level: str='normal'
    ) -> EnhancedMol:
        if emol.num_conformers == 0:
            emol = emol.init_geometry('openbabel')
        with run_in(self.workdir):
            xyz_path = emol.write_xyz('original.xyz')
            charge = emol.charge
            num_unpairs = emol.num_unpairs
            cmd = [self.xtb_path, xyz_path, '--silent', 
                   '--opt', level, 
                   '--chrg', f'{charge}', 
                   '--uhf', f'{num_unpairs}']
            results = subprocess.run(cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True,)
            optimized_path = os.path.abspath('xtbopt.xyz')

            with open('charges', 'r') as f: 
                charges = f.read().splitlines()
                charges = list(map(float, charges))
            self._stdouts['optimize'] = results.stdout
            self._stderrs['optimize'] = results.stderr
        optimized_geometry = EnhancedMol.from_xyz(optimized_path,
                                                  sanitize=False).geometry
        optimized_emol = emol.update_geometry(optimized_geometry)
        optimized_emol.partial_charges = charges
        return optimized_emol
    
    @staticmethod
    def _parse_esp_dat(path: str) -> PointCloud:
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) == 0:
                return None
            positions = []
            values = []
            for line in lines:
                x, y, z, value = map(float, line.split())
                positions.append([x, y, z])
                values.append(value)
            positions = np.array(positions)
            values = np.array(values)
        return PointCloud(positions, values, name='xtb_esp')
    
    def eval(
        self, 
        emol: EnhancedMol,
        target: List[Literal['esp', 'density', 'molden']]=['esp'],
        timeout: int=10,
        refresh_workdir: bool=True,
        rebase_dir: str=None
    ) -> Dict[str, Any]:
        if refresh_workdir:
            shutil.rmtree(self.workdir)
            os.makedirs(self.workdir)
        
        xyz_path = 'input.xyz'
        cmd = [self.xtb_path, xyz_path, '--silent', 
                '--chrg', f'{emol.charge}',
                '--uhf', f'{emol.num_unpairs}']
        
        if 'esp' in target:
            cmd += ['--esp']
        if 'density' in target:
            cmd += ['--input', XTB_INP]
        if 'molden' in target:
            cmd += ['--input', XTB_INP, '--molden']
        
        output = {}
        with run_in(self.workdir):
            emol.write_xyz(xyz_path)
            try:
                results = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                self._stdouts['-'.join(target)] = results.stdout
                self._stderrs['-'.join(target)] = results.stderr
            except Exception as e:
                if self.verbose:
                    print(f'xtb esp calculation failed: {e}')
                if os.path.exists('xtb_esp.dat') and refresh_workdir:
                    output['esp'] = self._parse_esp_dat('xtb_esp.dat')
                if os.path.exists('density.cub') and refresh_workdir:
                    output['density'] = Grid.from_cube('density.cub')
                if os.path.exists('molden.input') and refresh_workdir:
                    with open('molden.input', 'r') as f:
                        output['molden'] = f.read()
                return output
            
            if 'esp' in target:
                output['esp'] = self._parse_esp_dat('xtb_esp.dat')
            if 'density' in target:
                output['density'] = Grid.from_cube('density.cub')
            if 'molden' in target:
                with open('molden.input', 'r') as f:
                    output['molden'] = f.read()
            
            if rebase_dir is not None:
                shutil.copytree(self.workdir, rebase_dir)
        return output