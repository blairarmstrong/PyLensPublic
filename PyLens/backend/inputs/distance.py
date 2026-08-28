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

    def compute(self, prev_links):
        """
        Computes the squared Euclidean distance between the output of incoming groups and their respective weights.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            af: Computed distance matrix.
        """
        input_matrix = af.ones(prev_links[0].incoming_group.num_units)

        for link in prev_links:
            forward_output = (
                (link.outgoing_group.output_matrix - link.weights) ** 2).flatten()
            input_matrix += forward_output
            self.unitHistoryData = input_matrix

        return input_matrix

    def backward(self, prev_links, input_derivs):
        """
        Computes the derivative of the distance function and updates the weights accordingly.

        Args:
            prev_links (list): List of links connecting to the current group.
            input_derivs: Derivatives of inputs affecting weight updates.

        Returns:
            Updated output derivatives of the outgoing group.
        """
        delta = 0
        if input_derivs != 0:
            for link in prev_links:
                # act.c: real inputDeriv = U->inputDeriv * 2.0;
                input_deriv = input_derivs * 2
                # act.c: delta = inputDeriv * (L_WGT - V_OUT);
                delta = input_deriv * \
                    (link.weights - link.outgoing_group.output_matrix)
                # act.c: V_DRV -= delta;
                # act.c: L_DRV += delta;
                link.weights += delta

            if link.outgoing_group.group_type != "bias":
                link.outgoing_group.output_matrix -= delta
            else:
                link.outgoing_group.output_derivs -= [
                    sum((delta).flatten())]
        return link.outgoing_group.output_derivs
