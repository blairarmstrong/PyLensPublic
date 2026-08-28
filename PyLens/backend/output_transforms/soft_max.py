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

    def forward(self, x):
        return self.func(x)

    def backward(self, x, output_derivs):
        exp_x = af.exp(x)
        s = exp_x / af.sum(exp_x)
        deriv = -s[:,af.newaxis] @ af.array([s])
        af.fill_diagonal(deriv, s*(1-s))
        result = deriv @ output_derivs
        return result
