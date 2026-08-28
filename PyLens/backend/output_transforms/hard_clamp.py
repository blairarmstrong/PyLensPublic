from ..clamping import Clamping
from ..array_factory import Array_factory as af

class Hard_Clamp(Clamping):
    """
    If the externalInput is a real number, this sets the output to the externalInput. Otherwise it does nothing.
    """

    def __init__(self, group):
        super().__init__("Hard_clamp", group)

    def forward(self, x):
        # x = self.group.external_input
        x = af.where(af.isnan(self.group.external_input), x, self.group.external_input)
        return x

    def backward(self, x, output_derivs):
        ## External Input History = x
        # return input deriv of 0s
        return af.zeros(x.shape)
