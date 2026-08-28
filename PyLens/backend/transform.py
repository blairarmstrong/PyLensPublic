import os
from .array_factory import Array_factory as af

class Transform:

    name = None
    group = None

    
    def __init__(self, name, group):
        self.name = name
        self.baseType = os.getenv('BASETYPE')
        self.unitHistoryData = af.empty(((group.time_intervals * group.ticks_per_interval)+1, group.num_units))
        self.unitData = af.zeros(group.num_units)
        self.group = group
        
    def _read_group_field(self, group, field: str):
        candidates = {
            "inputs":         ["input_matrix"],
            "externalInputs": ["external_input"],
            "outputs":        ["output_matrix"],
            "targets":        ["target"],
            "inputDerivs":    ["input_derivs"],
            "outputDerivs":   ["output_derivs"],
        }
        for name in candidates.get(field, []):
            if hasattr(group, name):
                return getattr(group, name)
        raise AttributeError(f"Field '{field}' not found on group '{group.name}'.")

