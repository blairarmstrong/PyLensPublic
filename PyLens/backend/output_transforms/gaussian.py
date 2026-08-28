from .basic import Basic
from ..array_factory import Array_factory as af


class Gaussian(Basic):
    """This computes a gaussian radial basis function: exp(-i^2 * gain^2). This is often as effective as LOGISTIC,
    although it can become a bit unstable at the end of training. It can also be used in conjunction with ADAPTIVE_GAIN
    for individual, trainable gains for each unit.
    """
    gain = 1
    #func_deriv = None

    def __init__(self, group, gain=1):
        super().__init__("Gaussian", group)
        # makes sure gain takes on the right value,network or group or unit(array of gains).
        # Always set gain to an numpy array
        self.gain = [self.group.network.gain] if af.isnan(self.group.gain).any() else \
            (self.group.gain if isinstance(self.group.gain, list) else [self.group.gain])
        self.gain = af.array(self.gain)
       # self.func_deriv = elementwise_grad(self.func)

    def func(self, x):
        z = self.gain * x
        return af.exp(-z * z)
        # return 1 / (1 + np.exp(-(x ** 2) * (self.gain ** 2)))

    def forward(self, x):
        return self.func(x)

    def backward(self, x, output_derivs):
        scale = -2 * self.gain * self.gain
        # x is the input
        return output_derivs * self.group.output_history[self.group.curr_tick] * x * scale
        # return self.func_deriv(x)
