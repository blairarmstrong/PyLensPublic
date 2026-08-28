from typing import Union

from .input_transform import Input_Transform
from ..array_factory import Array_factory as af
from ..link.link import Link

import numpy as np

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
        # if ((self.group.external_input is None and self.group.target is None) 
        #      or not self.group.network.in_grace_period):
        #     for link in prev_links:
        #         input_matrix += af.dot(link.outgoing_group.output_matrix, link.weights)
        for i in range(self.group.num_units):
            if np.isnan(self.group.external_input[i]) and ( 
                np.isnan(self.group.target[i]) or not self.group.network.in_grace_period
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
            # Reverse sign for hebbian learning update because of optimizer weight substraction
            link.weight_derivs += link.outgoing_group.output_matrix[:, np.newaxis] @ link.incoming_group.output_matrix[np.newaxis, :]   \
                                  - link.outgoing_group.output_derivs[:, np.newaxis] @ link.incoming_group.output_derivs[np.newaxis, :]
