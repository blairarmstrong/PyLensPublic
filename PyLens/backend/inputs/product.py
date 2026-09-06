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

    def forward(self):
        """
        Computes the element-wise product between the output of incoming groups and their corresponding weights.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            af: Computed product matrix.
        """

        af.fill(self.group.input_matrix, 1)

        for link in self.group.incoming_links:
            self.group.input_matrix *= (
                link.outgoing_group.output_matrix
                * link.weights
            ).flatten()

        self.unitHistoryData[self.group.curr_tick] = (
            self.group.input_matrix
        )


    def backward(self):
        """
        Computes the derivative of the product for each linked group and updates weight derivatives.

        """
        input_derivs = self.group.input_derivs
        input_store = self.unitHistoryData[self.group.curr_tick]

        for link in self.group.incoming_links:
            p = input_derivs * input_store
            v = p / (
                link.outgoing_group.output_matrix
                * link.weights
            )

            if link.outgoing_group.group_type != "bias":
                link.outgoing_group.outputderivCache += (
                    v * link.weights
                )
            else:
                link.outgoing_group.outputderivCache += [
                    af.sum((v * link.weights).flatten())
                ]

            link.backward_prod(input_derivs, v)
