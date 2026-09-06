from ..clamping import Clamping
from ..array_factory import Array_factory as af


class Weak_Clamp(Clamping):


    def __init__(self, group):
        super().__init__("Weak_Clamp", group)

    def forward(self):

        tick = self.group.curr_tick

        self.unitHistoryData[tick] = self.group.output_matrix

        strength = (
            self.group.network.clamp_strength
            if af.isnan(self.group.clamp_strength)
            else self.group.clamp_strength
        )

        for i in range(self.group.num_units):
            if not af.isnan(self.group.external_input[i]):

                self.group.output_matrix[i] += strength * (
                    self.group.external_input[i]
                    - self.group.output_matrix[i]
                )


    def backward(self):
        strength = self.group.network.clamp_strength if af.isnan(self.group.clamp_strength) else self.group.clamp_strength
        scale = 1 - strength

        original_output = self.unitHistoryData[self.group.curr_tick]

        self.group.output_derivs[...] = af.where(
            self.group.output_matrix != original_output,
            self.group.output_derivs * scale,
            self.group.output_derivs
        )

        self.group.output_matrix[...] = original_output


