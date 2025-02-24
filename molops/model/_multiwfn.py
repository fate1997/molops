import codecs
import os
import subprocess
from typing import Dict, Literal

import pandas as pd

from molops.utils.facility import run_in

DEFAULT_EXEC_PATH = 'Multiwfn'

class MultiWfnModel:
    def __init__(
        self, 
        exec_path: str=DEFAULT_EXEC_PATH,
        work_dir: str='./multiwfn_workdir',
        verbose: bool=True
    ):
        self.exec_path = exec_path
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = os.path.abspath(work_dir)
        self.verbose = verbose
        self._stdouts = {}
    
    def get_mwinp(self, target: Literal['esp-hist'], **kwargs) -> str:
        """Get the input for Multiwfn."""
        if target == 'esp-hist':
            minima = kwargs.get('minima', -100)
            maxima = kwargs.get('maxima', 100)
            num_bins = kwargs.get('num_bins', 100)
            if self.verbose:
                print(f'esp-hist config: minima={minima}, maxima={maxima}, num_bins={num_bins}')
            mwinp = ['12', '3', '0.15', '0', '1', '9', 'all', 
                     f'{minima},{maxima}',
                     f'{num_bins}', '3']
        return mwinp
    
    def parse_esp_hist(self, output_path: str) -> Dict[float, float]:
        """Parse the output of Multiwfn esp-hist."""
        with codecs.open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print(lines)
            start_idx = None
            end_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('Begin'):
                    start_idx = i
                if line.strip().startswith('Sum'):
                    end_idx = i
                    break
            if start_idx is None or end_idx is None:
                return 
            columns = lines[start_idx].split()
            df_lines = lines[start_idx+1:end_idx]
            df_rows = [line.split() for line in df_lines]
            df = pd.DataFrame(df_rows, columns=columns, dtype=float)
        return df
    
    def get_esp_hist(
        self, 
        density_path: str,
        minima: float=None,
        maxima: float=None,
        num_bins: int=None
    ) -> pd.DataFrame:
        """Get the electrostatic potential histogram."""
        density_path = os.path.abspath(density_path)
        config = {'minima': minima, 'maxima': maxima, 'num_bins': num_bins}
        config = {k: v for k, v in config.items() if v is not None}
        with run_in(self.work_dir):
            mwinp = self.get_mwinp('esp-hist', **config)
            mwinp = '\n'.join(mwinp)
            mwinp_path = 'mw.inp'
            with open(mwinp_path, 'w') as f:
                f.write(mwinp)
            
            cmd = f'{self.exec_path} {density_path} <{mwinp_path}> esp_hist.out'
            #! Not working properly
            output = subprocess.run(cmd, capture_output=True, shell=True)
            if self.verbose:
                print(output.stdout.decode())
                print(output.stderr.decode())
            esp_hist = self.parse_esp_hist('esp_hist.out')
        return esp_hist
    
    def eval(
        self, 
        density_path: str,
        target: Literal['esp-hist']
    ):
        pass