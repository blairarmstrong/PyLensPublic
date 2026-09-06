from ..clamping import Clamping
from ..array_factory import Array_factory as af

class Bias_Clamp(Clamping):

    def __init__(self, group):
        super().__init__("Bias_Clamp", group)

    def forward(self):
        # TODO change this to be manipulated by user or to be bias.initOutput
        af.fill(self.group.output_matrix, 1)

    def backward(self):
        self.group.input_derivs[...] = 0
