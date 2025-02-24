from typing import Any, Dict

import numpy as np
from rdkit import Chem

from molops.constants import BOHR_2_ANGSTROM


def parse_cube(path: str) -> Dict[str, Any]:
    """Parse molecule from cube file."""
    with open(path) as f:
        lines = f.readlines()

    # Parse atom_num and geometry
    tot_atom_num = int(lines[2].split()[0])
    atom_num_lst = []
    electron_num_list = []
    geometry_lst = []
    for line in lines[6:6 + tot_atom_num]:
        line = line.split()
        atom_num_lst.append(int(line[0]))
        electron_num_list.append(float(line[1]))
        geometry_lst.append([eval(x) for x in line[2:5]])

    atom_num = np.asarray(atom_num_lst, dtype=int)
    electron_num = np.asarray(electron_num_list, dtype=int)
    geometry = np.asarray(geometry_lst, dtype=float)
    geometry = geometry * BOHR_2_ANGSTROM

    # Calculate partial charges
    charge_array = atom_num - electron_num
    
    # Generate XYZ string
    xyz = ''
    xyz += f'{tot_atom_num}\n\n'
    for i in range(tot_atom_num):
        atom_symbol = Chem.Atom(atom_num[i].item()).GetSymbol().ljust(2)
        x, y, z = geometry[i].tolist()
        xyz += f'{atom_symbol}   {x:>12.4f}{y:>12.4f}{z:>12.4f}\n'

    return {
        'xyz': xyz,
        'partial_charges': charge_array
    }