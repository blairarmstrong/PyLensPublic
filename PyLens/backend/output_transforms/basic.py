from .activation import Activation


class Basic(Activation):
    """
    This class implements the output transformation (activation) from a group of units. It serves as the base for different kinds of output transformations.
    """

    def __init__(self, name, group):
        super().__init__(name, group)
