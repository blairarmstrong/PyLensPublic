from typing import Union

from .input_transform import Input_Transform
from ..array_factory import Array_factory as af
from ..link.link import Link

class BoltzmannInput(Input_Transform):
    def __init__(self, group: "Group"):
        super().__init__("BoltzmannInput", group)

    def forward(self):
        """
        Computes the input to Boltzmann machine units.

        Args:
            prev_links (Union[list, Link]): List of links incoming to the current group.
        
        Returns:
            af: Computed input matrix.
        """
        af.fill(self.group.input_matrix, 0)

        if self.group.network.in_grace_period:
            free = (
                af.isnan(self.group.external_input)
                & af.isnan(self.group.target)
            )
        else:
            free = af.isnan(self.group.external_input)

        for link in self.group.incoming_links:
            self.group.input_matrix += (
                link.outgoing_group.output_matrix @ link.weights
            )

        self.group.input_matrix[~free] = 0

    def backward(self):
        """
        Computes and sums Hebbian learning updates to link derivatives.

        Args:
            prev_links (Union[list, Link]): List of links incoming to the current group.
            *args: Additional arguments for the backward computation.
        """
        for link in self.group.incoming_links:
            source = link.outgoing_group
            # Reverse sign for Hebbian learning update because of optimizer weight subtraction
            # compute outer product for coactivation in hebbian learning
            # output_derivs stores the outputs in the positive phase
            link.weight_derivs += (
                source.output_matrix[:, None]
                @ self.group.output_matrix[None, :]
                - source.output_derivs[:, None]
                @ self.group.output_derivs[None, :]
            )
