import os
from ..transform import Transform

class Input_Transform(Transform):
    """
    This class implements the input transformation to a group of units. It serves as the base for different kinds of input transformations.
    """

    def __init__(self, name, group):
        super().__init__(name, group)

    def forward(self):
        pass

    def backward(self):
        pass
