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
        self.unitData = af.array(self.group.num_units)
        self.dt = self.group.network.dt * self.group.dt

    def forward(self, x):
        last_output = self.unitData
        dt = self.dt
        min = self.group.minOutput
        max = self.group.maxOutput
        ## Idk whats this, but somehow 0 works for clens, maybe incorrect initialization
        # rest = self.group.initOutput
        rest = 0
        dt_scale = 1 ### What is this? clens : U->dtScale
        input = x
        output = last_output

        input_flag = af.where(input > 0, 1, 0)

        input_pos = input * input_flag
        input_neg = input * ((1 - input_flag)*1)

        output = output + dt * dt_scale * ( ((max - output) * input_pos + (output - min) * input_neg) - (output - rest))


        output = af.where( output > max, max, output )
        output = af.where( output < 0, 0, output )
        self.unitData = copy.copy(output)
        # self.group.output_matrix = copy.copy(output)
        return output

    # No Backward Pass for Interact Integr
