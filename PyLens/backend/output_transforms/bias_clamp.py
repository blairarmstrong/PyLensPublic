from ..clamping import Clamping
import numpy as np
from ..array_factory import Array_factory as af
class Bias_Clamp(Clamping):

    def __init__(self, group):
        super().__init__("Bias_Clamp", group)

    def forward(self, x):
        # TODO change this to be manipulated by user or to be bias.initOutput
        output = np.ones(x.shape)
        return output

    def backward(self, x, output_derivs):
        # self.group.input_derivs = af.zeros(self.group.input_derivs.shape)
        return af.zeros(self.group.input_derivs.shape)
