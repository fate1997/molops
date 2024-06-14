# The implementation of lattice enthalpy calculation from:
# https://pubs.acs.org/doi/10.1021/ja0619612

from pint import UnitRegistry

from molops.emol import EnhancedMol

from .crystal_volume import get_crystal_volume
from .register import register_property

ureg = UnitRegistry()

GAS_CONSTANT = 8.314 * 10 ** -3 * ureg('kJ/(mol*K)')


@register_property('lattice_enthalpy')
def get_delta_latt_enthalpy(emol: EnhancedMol, 
                            temperature: float = 298.15, 
                            return_U_pot: bool = False):
    r"""Calculate the lattice enthalpy of a crystal."""
    volume = get_crystal_volume(emol)
    volume = volume * ureg('nm^3')
    temperature = temperature * ureg('K')
    alpha = 117.3 * ureg('kJ/mol')
    beta = 51.9 * ureg('kJ/mol')
    U_pot = 2 * (alpha / (volume / ureg('nm^3')) ** (1/3) + beta)
    H_latt = U_pot + 2 * GAS_CONSTANT * temperature
    if return_U_pot:
        return H_latt.magnitude, U_pot.magnitude
    return H_latt.magnitude


if __name__ == '__main__':
    import pandas as pd
    test_cases = pd.read_csv('test_cases.csv')
    for idx, row in test_cases.iterrows():
        volume = row.molecular_volume
        expected = row.delta_latt_H * ureg('kJ/mol')
        result = get_delta_latt_enthalpy(volume)
        print(f'Case {idx + 1}: {result:.3f} (expected: {expected:.3f})')