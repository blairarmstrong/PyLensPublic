from .input_transform import Input_Transform
from ..array_factory import Array_factory as af

class Dot_Product(Input_Transform):
    """
    Computes the dot product of all incoming groups and their respective weights.
    """

    def __init__(self, group):
        """
        Initializes the Dot_Product transformation.

        Args:
            group: The neural network group to which this transformation is applied.
        """
        super().__init__("dot", group)

    def forward(self):
        """
        Computes the dot product between the output of the outgoing groups and their corresponding weights.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            af: Computed dot product matrix.
        """
        af.fill(self.group.input_matrix, 0)

        for link in self.group.incoming_links:
            self.group.input_matrix += link.forward(
                link.outgoing_group.output_matrix
            )

    def backward(self):
        """
        Computes the derivative of the dot product for each linked group and updates weight derivatives.

        """
        input_derivs = self.group.input_derivs

        for link in self.group.incoming_links:
            link.outgoing_group.outputderivCache += input_derivs @ link.weights.T
            link.backward(self.group.input_derivs)
