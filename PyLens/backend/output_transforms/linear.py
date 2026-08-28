from .basic import Basic


class Linear(Basic):
    """This simply copies the input to the output.
    """
    func_deriv = None

    def __init__(self, group):
        super().__init__("Linear", group)
        self.func_deriv = elementwise_grad(self.func)

    def func(self, x):
        return x

    def forward(self, x):
        return self.func(x)

    def backward(self, x, output_derivs):
        deriv = self.func_deriv(x)
        result = output_derivs * deriv
        return result
