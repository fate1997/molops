from pint import UnitRegistry
from .crystal_volume import get_crystal_volume
from molops.emol import EnhancedMol
from .register import register_property

ureg = UnitRegistry()


@register_property('solid_entropy')
def get_solid_entropy(emol: EnhancedMol):
    volume = get_crystal_volume(emol)
    volume = volume * ureg('nm^3')
    k = 1360 * ureg('J/(mol*K*nm^3)')
    c = 15 * ureg('J/(mol*K)')
    solid_entropy_298 = (k * volume + c) * 10 ** -3
    return solid_entropy_298.magnitude


if __name__ == '__main__':
    solid_entroies = {
        0.229: 0.3264,
        0.287: 0.4053,
        0.388: 0.5427
    }
    for volume, entropy in solid_entroies.items():
        calculated_entropy = get_solid_entropy(volume)
        print(f'{volume} nm^3 solid entropy: {calculated_entropy:.4f} (expected: {entropy:.4f})')