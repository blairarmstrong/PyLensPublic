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

    def compute(self, prev_links):
        """
        Computes the dot product between the output of the outgoing groups and their corresponding weights.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            af: Computed dot product matrix.
        """
        input_matrix = af.zeros(prev_links[0].incoming_group.num_units)
        for link in prev_links:
            forward_output = link.forward(link.outgoing_group.output_matrix)

            input_matrix += forward_output

        return input_matrix

    def backward(self, prev_links, input_derivs):
        """
        Computes the derivative of the dot product for each linked group and updates weight derivatives.

        Args:
            prev_links (list): List of links connecting to the current group.
            input_derivs (np.array): Derivatives of the inputs affecting weight updates.
        """
        for link in prev_links:
            if link.outgoing_group.network.parallel_mode:
                link.outgoing_group.increment_outputderiveCache(input_derivs @ link.weights.T)
            else:
                link.outgoing_group.outputderivCache += input_derivs @ link.weights.T
            
            link.backward(input_derivs)
