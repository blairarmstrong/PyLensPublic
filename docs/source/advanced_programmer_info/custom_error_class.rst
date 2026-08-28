Custom Error Class
===================

The loss function will involve building a class that implements the Error class. In init function, call super().__init__("loss name") to initialize with error function's name. If there are additional parameters, initialize them near class header.

If you would like to have autograd to calculate derivative automatically, write "self.func_deriv = elementwise_grad(self.func)" in the init method to have that automation setup.

Write your forward operation in self.func method, and write the backward operation in self.backward method. If you are using autograd, you just need to write "return self.func_deriv(outputs, targets)", If for some reason the autograd cannot properly calculate the loss, you will have to manually put in the formula for calculating derivatives.

For example:

.. code-block:: python

    class MeanSquareError(Error):
        func_deriv = None
        def __init__(self):
            super().__init__("Mean Squared Error")

            # take the derivative of the function to use in the backward pass
            self.func_deriv = elementwise_grad(self.func)
        def func(self, outputs, targets):
            return (targets - outputs)*(targets - outputs)
        def forward(self, outputs, targets):
            return self.func(outputs, targets)
        def backward(self, outputs, targets):
            return self.func_deriv(outputs, targets)

To use the custom loss function, you will need to pass it into the error_function argument when adding output group to the network.

For example:

.. code-block:: python

    my_error = MeanSquareError()
    ...
    xor_net.add_group(1, name="output", group_type="output", error_function=my_error)