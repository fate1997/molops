from molops.emol import GeometryOptimizer, EnhancedMol


PROPERTY_REGISTRY = {}

def register_property(name: str):
    def decorator(property_calculator):
        PROPERTY_REGISTRY[name] = property_calculator
        return property_calculator
    return decorator

def get_property_calc(name: str):
    return PROPERTY_REGISTRY[name]

def get_available_properties():
    return list(PROPERTY_REGISTRY.keys())

def require_geometry(func):
    def wrapper(emol, *args, **kwargs):
        if emol.rdmol.GetNumConformers() == 0:
            print(f'No geometry available for {emol.smiles}, will generate one by ETKDG')
            mol = GeometryOptimizer._initialize_by_rdkit(emol)
            emol = EnhancedMol(mol, emol.smiles)
        return func(emol, *args, **kwargs)
    return wrapper