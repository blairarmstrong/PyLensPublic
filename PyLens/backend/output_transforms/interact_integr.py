from .basic import Basic
from ..array_factory import Array_factory as af
import copy


class Interact_Integr(Basic):
    """This is just exp(i). There is a big potential for overflow with this, so you may want to be careful how you
    use it.
    """
    func_deriv = None

    def __init__(self, group):
        super().__init__("Interact_integr", group)
        self.unitData = af.zeros(self.group.num_units)
        self.dt = self.group.network.dt * self.group.dt

    def forward(self):
        input_matrix = self.group.input_matrix
        output = self.unitData

        dt = self.dt
        min_output = self.group.minOutput
        max_output = self.group.maxOutput
        rest = 0
        dt_scale = 1

        input_flag = af.where(input_matrix > 0, 1, 0)

        input_pos = input_matrix * input_flag
        input_neg = input_matrix * (1 - input_flag)

        output = output + dt * dt_scale * (
            (
                (max_output - output) * input_pos
                + (output - min_output) * input_neg
            )
            - (output - rest)
        )

        output = af.where(output > max_output, max_output, output)
        output = af.where(output < 0, 0, output)

        self.unitData[...] = output
        self.group.output_matrix[...] = output

    # No Backward Pass for Interact Integr
