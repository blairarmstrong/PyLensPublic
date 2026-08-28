import os
from ..transform import Transform

class Activation(Transform):

    x = None

    def __init__(self, name, group):
        super().__init__(name, group)

    def forward(self, x):
        return x
