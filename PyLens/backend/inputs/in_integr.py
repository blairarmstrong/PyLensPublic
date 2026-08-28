from .input_transform import Input_Transform

class In_Integr(Input_Transform):
    """
    Integrates the output rather than the input in a continuous network.
    This transformation is applied by default unless `IN_INTEGR` is explicitly specified.
    """

    def __init__(self, group):
        """
        Initializes the In_Integr transformation.

        Args:
            group: The neural network group to which this transformation is applied.
        """
        super().__init__("In_Integr", group)
        self.dt = self.group.network.dt * self.group.dt

    def compute(self, prev_links):
        """
        Computes the integrated output by applying a time-based update rule.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            np.array: Updated input matrix after integration.
        """
        lastinput = self.unitData
        lastinput += self.dt * (self.group.input_matrix - lastinput)
        self.group.input_matrix = lastinput
        return lastinput


    def backward(self, prev_links, input_derivs):
        """
        Computes the backward pass for the integration transformation.

        Args:
            prev_links (list)
            input_derivs (np.array): Derivatives of the input values.

        Returns:
            np.array: Updated derivative values after integration.
        """
        lastinputderiv = self.unitData
        self.unitData += self.dt * (input_derivs - lastinputderiv)
        self.group.input_derivs = lastinputderiv
        return self.unitData
