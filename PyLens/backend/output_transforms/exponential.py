from .basic import Basic
from ..array_factory import Array_factory as af

class Exponential(Basic):
    """This is just exp(i). There is a big potential for overflow with this, so you may want to be careful how you
    use it.
    """
    func_deriv = None

    def __init__(self, group):
        super().__init__("Exponential", group)
        # self.func_deriv = grad(self.func)

    def func(self, x):
        return af.exp(x)

    def forward(self):
        self.group.output_matrix[...] = af.exp(
            self.group.input_matrix
        )

    def backward(self):
        self.group.input_derivs[...] = (
            self.group.output_matrix
            * self.group.output_derivs
        )
