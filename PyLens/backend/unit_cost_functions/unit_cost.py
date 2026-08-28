import math

class UnitCost(Exception):
    """
    Represents a base class for unit cost, designed to be extended by specific cost functions.

    Parameters:
        name (str): The name of the unit cost.
        group (str): The group that the unit cost belongs to.

    Methods:
        forward(outputs): Performs the forward pass of the unit cost.
        backward(outputs): Performs the backward pass of the unit cost.
        choose_value(a, b): Chooses a value between `a` and `b`.

    Attributes:
        name (str): The name of the unit cost.
        group (str): The group of the unit cost.
    """

    def __init__(self, name, group):
        self.name = name
        self.group = group

    def forward(self, outputs):
        return None

    def backward(self, outputs):
        return None
    
    def choose_value(a, b):
        """
        Chooses between two values `a` and `b` based on whether `a` is NaN.

        This utility function returns `a` if it is a valid (non-NaN) value, and returns `b`
        if `a` is NaN. It can be useful in situations where default values need to be applied
        when certain parameters are undefined (NaN).

        Parameters:
            a: The first value to choose from, which may be NaN.
            b: The second value, which is used as a fallback if `a` is NaN.

        Returns:
            The value `a` if it is not NaN, otherwise the value `b`.
        """
        if math.isnan(a):
            return b
        else:
            return a


