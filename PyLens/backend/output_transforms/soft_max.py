from ..clamping import Clamping
from ..array_factory import Array_factory as af


class SoftMax(Clamping):
    """This is equivalent to an exponential followed by a normalization. However, SOFT_MAX scales the values before
    computing the exponential. This doesn't affect the end result but it avoids overflow. A SOFT_MAX OUTPUT group will
    get DIVERGENCE error by default.
    """
    func_deriv = None

    def __init__(self, group):
        super().__init__("soft_max", group)

    def func(self, x):
        exp_x = af.exp(x - af.max(x))
        return exp_x / af.sum(exp_x)

    def forward(self):
        self.group.output_matrix[...] = self.func(
            self.group.input_matrix
        )

    def backward(self):
        output = self.group.output_matrix
        output_derivs = self.group.output_derivs

        output_deriv_sum = af.sum(output_derivs * output)

        self.group.input_derivs[...] = (
            output * (output_derivs - output_deriv_sum)
        )
