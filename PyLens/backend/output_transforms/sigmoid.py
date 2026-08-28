from .basic import Basic
from ..array_factory import Array_factory as af


class Sigmoid(Basic):
   # gain = 1
    func_deriv = None

    def __init__(self, group, gain=1):
        super().__init__("Sigmoid", group)
        #self.gain = gain
        self.gain = [self.group.network.gain] if af.isnan(self.group.gain).any() else \
            (self.group.gain if isinstance(self.group.gain, list) else [self.group.gain])
        self.gain = af.array(self.gain)
        # self.func_deriv = elementwise_grad(self.func) #jacobian(self.func)

    """
    Instantiates the Sigmoid function for passing data fwd and back

    :param x: the values to pass into the function
    :type x: ndarray
    """
    def func(self, x):
        # Scale the input by gain and flip the sign (for sigmoid)
        # x = self.group.input_matrix
        x_scaled = -x * self.gain

        # Clip input to avoid overflow in np.exp:
        # np.exp(709) is approximately 8.2e+307 (still within float64 range),
        # but np.exp(710) overflows to inf.
        # So we clip to [-709, 709] to stay within the safe range for float64 exponentials.
        x_scaled = af.clip(x_scaled, -709, 709)

        return 1 / (1 + af.exp(x_scaled))
            

    """
    Passes data forward into the sigmoid transform

    :param x: the values to pass in 
    :type x: ndarray
    """
    def forward(self, x):
        return self.func(x)

    """
    Passes data backward into the deriv of the sigmoid transform

    :param x: the values to pass in 
    :type x: ndarray
    """
    def backward(self, x, output_derivs):
        # deriv = self.func_deriv(x)
        deriv = self.func(x) * (1- self.func(x)) #manual deriv
        result = output_derivs * deriv
        return result
