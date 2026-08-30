from ..clamping import Clamping
import copy
from ..array_factory import Array_factory as af


class Weak_Clamp(Clamping):


    def __init__(self, group):
        super().__init__("Weak_Clamp", group)

    def forward(self, x):
        self.unitHistoryData[self.group.curr_tick] = copy.copy(x)

        strength = (
            self.group.network.clamp_strength
            if af.isnan(self.group.clamp_strength)
            else self.group.clamp_strength
        )

        for i in range(self.group.num_units):
            if not af.isnan(self.group.external_input[i]):
                x[i] += strength * (
                    self.group.external_input[i] - x[i]
                )

        self.current_output = copy.copy(x)

        return x

    def backward(self, x, output_derivs):
        strength = self.group.network.clamp_strength if af.isnan(self.group.clamp_strength) else self.group.clamp_strength
        scale = 1 - strength

        original_output = self.unitHistoryData[self.group.curr_tick]

        self.group.output_derivs = af.where(
            original_output != self.current_output,
            output_derivs * scale,
            output_derivs
        )

        return self.group.input_derivs

