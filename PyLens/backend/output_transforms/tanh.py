from .basic import Basic
from .sigmoid import Sigmoid
from ..array_factory import Array_factory as af

class Tanh(Basic):
    """This is equivalent to 1 - 2S(2 i), where S is the ordinary sigmoid function and I is the input. Note that its
    slope is actually twice what the slope would be if you just stretched a sigmoid to the range [-1,1]. So you may
    want to use half the normal gain to compensate. If ADAPTIVE_GAIN is used, each unit will have its own trainable
    gain.
    """
    func_deriv = None
    #gain = 1

    def __init__(self, group, gain=1):
        super().__init__("Tanh", group)
        self.gain = [self.group.network.gain] if af.isnan(self.group.gain).any() else \
            (self.group.gain if isinstance(self.group.gain, list) else [self.group.gain])
        self.gain = af.array(self.gain)
        self.sig = Sigmoid(group)

    def forward(self):
        self.group.output_matrix[...] = af.tanh(
            self.gain * self.group.input_matrix
        )

    def backward(self):
        output = self.group.output_matrix

        self.group.input_derivs[...] = (
            self.group.output_derivs
            * (1 - output**2)
            * self.gain
        )
