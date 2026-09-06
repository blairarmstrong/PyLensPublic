from .input_transform import Input_Transform
from ..array_factory import Array_factory as af


class Distance(Input_Transform):
    """
    Computes the squared distance between the output of incoming groups and their respective weights.
    """
    def __init__(self, group):
        """
        Initializes the Distance transformation.

        Args:
            group: The neural network group to which this transformation is applied.
        """
        super().__init__("distance", group)

    def forward(self):
        """
        Computes the squared Euclidean distance between the output of incoming groups and their respective weights.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            af: Computed distance matrix.
        """
        af.fill(self.group.input_matrix, 0)

        for link in self.group.incoming_links:
            source = link.outgoing_group

            self.group.input_matrix += af.sum(
                (link.weights - source.output_matrix[:, None]) ** 2,
                axis=0
            )


    def backward(self):
        """
        Computes the derivative of the distance function and updates the weights accordingly.

        Args:
            prev_links (list): List of links connecting to the current group.
            input_derivs: Derivatives of inputs affecting weight updates.

        Returns:
            Updated output derivatives of the outgoing group.
        """
        input_derivs = self.group.input_derivs

        for link in self.group.incoming_links:
            source = link.outgoing_group

            delta = (
                2
                * (link.weights - source.output_matrix[:, None])
                * input_derivs[None, :]
            )

            source.outputderivCache -= af.sum(delta, axis=1)
            link.weight_derivs += delta
