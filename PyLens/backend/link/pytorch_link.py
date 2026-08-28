from .link import Link
import torch
import numpy as np


class PytorchLink(Link):
    weights: torch.Tensor
    weight_derivs: torch.Tensor
    lesion_mask: torch.Tensor
    last_weight_delta: torch.Tensor

    def __init__(self, outgoing_group, incoming_group, weights, link_type='exhibitory', dropout_rate=None, perma_lesion_rate=None):
        super().__init__(outgoing_group, incoming_group, weights, link_type, dropout_rate,
                         perma_lesion_rate)

        self.weight_derivs = torch.zeros(self.weights.shape)
        self.last_weight_delta = torch.zeros(self.weights.shape)

    def forward(self, output_unit):
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.connection_drop_out(self.dropout_rate)
        else:
            self.dropout_mask = None
        result_weight = self.weights
        if self.dropout_mask is not None:
            result_weight = self.dropout_mask.multiply(result_weight).A if type(
                self.dropout_mask) is not np.ndarray else self.dropout_mask * (result_weight)
        if self.lesion_mask is not None:
            result_weight = self.lesion_mask.multiply(result_weight).A if type(
                self.lesion_mask) is not np.ndarray else self.lesion_mask * (result_weight)
        result = torch.matmul(output_unit, result_weight)

        return result


class PytorchLinkFull(PytorchLink):
    def __init__(self, outgoing_group, incoming_group, weights, dropout_rate=None,
                 perma_lesion_rate=None):
        super().__init__(outgoing_group, incoming_group, weights, dropout_rate=dropout_rate,
                         perma_lesion_rate=perma_lesion_rate)

    @classmethod
    def uniform(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None,
                perma_lesion_rate=None):
        """
        Initializes weights using a uniform random distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            mean (int): Mean value for weight initialization.
            range (int): Range for weight initialization.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            PytorchLinkFull: A new instance of PytorchLinkFull.
        """
        weights = torch.empty(outgoing_group.num_units, incoming_group.num_units).uniform_(mean - range,
                                                                                           mean + range).requires_grad_()
        return PytorchLinkFull(outgoing_group, incoming_group, weights, dropout_rate, perma_lesion_rate)

    @classmethod
    def gaussian(cls, outgoing_group, incoming_group, mean, range, perma_lesion_rate=None):
        """
        Initialize weights from the gaussian random distribution

        :param cls: this is equivalent to self, creating a instantiation of this Link object
        :type cls: Link
        :param outgoing_group: the sending group
        :type outgoing_group: Group
        :param incoming_group: the receiving group
        :type incoming_group: Group
        :param mean: the mean
        :type mean: int
        :param range: the range/variance
        :type range: int
        """
        weights = torch.empty(outgoing_group.num_units, incoming_group.num_units).normal_(mean, range).requires_grad_()
        return PytorchLinkFull(outgoing_group, incoming_group, weights, dropout_rate, perma_lesion_rate)


class PytorchLinkFactory:
    @staticmethod
    def construct_link(outgoing_group, incoming_group, link_type, rand_mean=0, rand_range=1, proj_type="full",
                       dropout_rate=None, perma_lesion_rate=None):
        proj_types = {"full": PytorchLinkFull}
        link_class = proj_types[proj_type]
        if link_type == "uniform":
            return link_class.uniform(outgoing_group, incoming_group, rand_mean, rand_range, dropout_rate=dropout_rate,
                                      perma_lesion_rate=perma_lesion_rate)
        elif link_type == "gaussian":
            return link_class.gaussian(outgoing_group, incoming_group, rand_mean, rand_range,
                                       dropout_rate=dropout_rate, perma_lesion_rate=perma_lesion_rate)
        elif link_type == "kaiming":
            return link_class.kaiming_normal(outgoing_group, incoming_group, dropout_rate=dropout_rate,
                                             perma_lesion_rate=perma_lesion_rate)
