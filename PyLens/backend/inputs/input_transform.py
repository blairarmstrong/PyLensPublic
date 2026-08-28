import os
from ..transform import Transform

class Input_Transform(Transform):
    """
    This class implements the input transformation to a group of units. It serves as the base for different kinds of input transformations.
    """

    def __init__(self, name, group):
        """
        Initializes the Input_Transform class.

        Args:
            name (str): Name of the transformation.
            group: The neural network group to which this transformation is applied.
        """
        super().__init__(name, group)

    def compute(self, prev_links):
        """
        Abstract method for computing the input transformation.
        Must be implemented by subclasses.
        """
        pass
