from .input_transform import Input_Transform
from ..array_factory import Array_factory as af


class Product(Input_Transform):

    def __init__(self, group):
        """
        Initializes the Product transformation.

        Args:
            group: The neural network group to which this transformation is applied.
        """
        super().__init__("product", group)

    def compute(self, prev_links):
        """
        Computes the element-wise product between the output of incoming groups and their corresponding weights.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            af: Computed product matrix.
        """

        input_matrix = af.ones(prev_links[0].incoming_group.num_units)

        for link in prev_links:
            forward_output = (link.outgoing_group.output_matrix*link.weights).flatten()
            input_matrix *= forward_output
            self.unitHistoryData = input_matrix

        return input_matrix

    def backward(self, prev_links, input_derivs):
        """
        Computes the derivative of the product for each linked group and updates weight derivatives.

        Args:
            prev_links (list): List of links connecting to the current group.
            input_derivs (np.array): Derivatives of the inputs affecting weight updates.
        """

        for link in prev_links:

            p = input_derivs*self.unitHistoryData
            v = p/(link.outgoing_group.output_matrix*link.weights)

            if link.outgoing_group.group_type != "bias":
                link.outgoing_group.output_derivs += v*link.weights
            else:
                link.outgoing_group.output_derivs += [sum((v*link.weights).flatten())]
            link.backward_prod(input_derivs, v)
