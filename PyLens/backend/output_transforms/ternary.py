from .basic import Basic
from ..array_factory import Array_factory as af
from ..parameters import OptimizerParameters

params = OptimizerParameters()

class Ternary(Basic):
    """This is essentially a normal sigmoid shifted to the right added to a negated sigmoid of -i shifted to the left.
    Alternately, you can think of it as a [-1,1] sigmoid that has a flat place at 0. It is designed to give the unit
    stable outputs at -1, 1, and 0. You could think of such units as coding whether a feature is present, absent, or
    unknown. The gain affects the slope of each of the two sigmoids. The ternaryShift sets the distance between their
    centers. Increasing the ternaryShift will make the central plateau wider. Increasing the gain will make the
    transitions between plateaus sharper.
    """
    #gain = 1
    func_deriv = None

    def __init__(self, group):
        ## gain?
        super().__init__("Ternary", group)
        #self.gain = gain
        # self.sig = Sigmoid(group)
        # self.func_deriv = grad(self.func)

        # Unable to change ternary Shift
        # self.ternary_shift = params.PAR_O_ternaryShift
        # self.ternary_shift = 5
        # self.gain = 1

    def forward(self, x):

        gain = self.group.network.gain if af.isnan(self.group.gain) else self.group.gain
        ternary_shift = self.group.network.ternary_shift if af.isnan(self.group.ternary_shift) else self.group.ternary_shift
        y = af.exp(gain * ternary_shift)
        x_1 = af.exp(gain * x)
        z = x_1 * y
        return ((x_1 * z) - y) / ((x_1 + y) * (z + 1.0))
        # return self.func(x)

    def backward(self, x, output_derivs):

        gain = self.group.network.gain if af.isnan(self.group.gain) else self.group.gain
        ternary_shift = self.group.network.ternary_shift if af.isnan(self.group.ternary_shift) else self.group.ternary_shift
        y = af.exp(gain * ternary_shift)
        x_1 = af.exp(gain * x)
        z = x_1 * y
        v = af.square(x_1 + y)
        w = af.square(z + 1.0)
        return output_derivs * gain * z * (v + w) / (v * w)
        # return output_derivs
