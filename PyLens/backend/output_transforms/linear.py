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

    def forward(self):
        self.group.output_matrix[...] = self.group.input_matrix

    def backward(self):
        self.group.input_derivs[...] = self.group.output_derivs
