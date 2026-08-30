from typing import Union

from .input_transform import Input_Transform
from ..array_factory import Array_factory as af
from ..link.link import Link

class BoltzmannInput(Input_Transform):
    def __init__(self, group: "Group"):
        super().__init__("BoltzmannInput", group)

    def compute(self, prev_links: Union[list, Link]) -> af:
        """
        Computes the input to Boltzmann machine units.

        Args:
            prev_links (Union[list, Link]): List of links incoming to the current group.
        
        Returns:
            af: Computed input matrix.
        """
        input_matrix = af.zeros(prev_links[0].incoming_group.num_units) 

        for i in range(self.group.num_units):
            if af.isnan(self.group.external_input[i]) and ( 
                af.isnan(self.group.target[i]) or not self.group.network.in_grace_period
                ):
                for link in prev_links:
                    input_matrix[i] += af.dot(link.outgoing_group.output_matrix, link.weights[:, i])
        return input_matrix

    def backward(self, prev_links: Union[list, Link], *args) -> af:
        """
        Computes and sums Hebbian learning updates to link derivatives.

        Args:
            prev_links (Union[list, Link]): List of links incoming to the current group.
            *args: Additional arguments for the backward computation.
        """
        for link in prev_links:
            # Reverse sign for Hebbian learning update because of optimizer weight subtraction
            # compute outer product for coactivation in hebbian learning
            # output_derivs stores the outputs in the positive phase
            link.weight_derivs += (
                link.outgoing_group.output_matrix[:, None]
                @ link.incoming_group.output_matrix[None, :]
                - link.outgoing_group.output_derivs[:, None]
                @ link.incoming_group.output_derivs[None, :]
            )
