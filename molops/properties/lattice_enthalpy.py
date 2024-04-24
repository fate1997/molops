from pint import UnitRegistry
from .crystal_volume import get_crystal_volume
from molops.emol import EnhancedMol
from .register import register_property

ureg = UnitRegistry()

GAS_CONSTANT = 8.314 * 10 ** -3 * ureg('kJ/(mol*K)')


@register_property('lattice_enthalpy')
def get_delta_latt_enthalpy(emol: EnhancedMol, temperature: float = 298.15):
    volume = get_crystal_volume(emol)
    volume = volume * ureg('nm^3')
    temperature = temperature * ureg('K')
    alpha = 117.3 * ureg('kJ/mol')
    beta = 51.9 * ureg('kJ/mol')
    U_pot = 2 * (alpha / (volume / ureg('nm^3')) ** (1/3) + beta)
    H_latt = U_pot + 2 * GAS_CONSTANT * temperature
    return H_latt.magnitude


if __name__ == '__main__':
    import pandas as pd
    test_cases = pd.read_csv('test_cases.csv')
    for idx, row in test_cases.iterrows():
        volume = row.molecular_volume
        expected = row.delta_latt_H * ureg('kJ/mol')
        result = get_delta_latt_enthalpy(volume)
        print(f'Case {idx + 1}: {result:.3f} (expected: {expected:.3f})')