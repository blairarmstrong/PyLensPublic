import math

import numpy as np
from ..array_factory import Array_factory as af
from scipy.sparse import rand, csc_matrix
from ..parameters import LinkParameters
import os


# import inspect

class Link:
    """
    Links (weights) between different groups (layers) of neural network.
    Abstract class that represents different types links: LinkFull, LinkOneToOne, LinkRandom.

    """

    # :param weights: Weights of the specific link
    # :type weights: np.array
    # :param weight_derivs: Keep track of the gradient of the weights at current backward pass
    # :type weight_derivs: np.array
    # :param outgoing_group: The group (layer) that send this link out
    # :type outgoing_group: Group
    # :param incoming_group: The group (layer) that receives the link
    # :type incoming_group: Group
    # :param lesion_mask: The sparse mask/numpy array of 0 and 1 such that 0 means masking specific value.
    # :type lesion_mask: np.array
    # :param frozen: True if the link’s weight cannot be updated, False if link’s weight can be updated during backpropagation.
    # :type frozen: bool
    # :param alpha: Link level learning rate, this will override network level learning rate
    # :type alpha: float
    # :param last_weight_delta: Keep track of the gradient of the weights of previous backward pass
    # :type last_weight_delta: np.array
    # :param max_weights: An upper bound on the values in weight matrix
    # :type max_weights: float
    # :param min_weights: An lower bound on values in weight matrix
    # :type min_weights: float

    weights: object
    weight_derivs: object
    outgoing_group: "Group"
    incoming_group: "Group"
    lesion_mask: object
    frozen: bool
    alpha: float
    last_weight_delta: object
    max_weights: float
    min_weights: float
    initialization: str
    link_type: str | None

    weights = None
    weight_derivs = None
    outgoing_group = None
    incoming_group = None
    lesion_mask = None
    dropout_mask = None
    connection_mask = None
    frozen = False
    freeze_mask = None

    def __init__(self, outgoing_group, incoming_group, weights, initialization="uniform", link_type=None,
                 dropout_rate=None, perma_lesion_rate=None):
        self.baseType = os.getenv('BASETYPE')
        link_params = LinkParameters()
        self.outgoing_group = outgoing_group
        self.incoming_group = incoming_group
        self.alpha = 1
        self.weights = weights
        self.weight_derivs = af.zeros(self.weights.shape)
        self.last_weight_delta = af.zeros(self.weights.shape)
        self.max_weights = link_params.PAR_L_max_weights
        self.min_weights = link_params.PAR_L_min_weights
        self.initialization = initialization
        self.link_type = link_type
        self.dropout_rate = dropout_rate
        self.perma_lesion_rate = perma_lesion_rate
        self.pre_noise_weight = None
        self.noise_mask = None
        self.freeze_mask = None

    def freeze(self):
        """Operation to freeze the weight"""
        self.frozen = True

    def unfreeze(self):
        """Operation to unfreeze the weight"""
        self.frozen = False

    def lesion_link(self, p):
        """
        Randomly lesions a percentage `p` of the connections in the link.

        Args:
            p (float): Probability of lesioning a connection (between 0 and 1).
        """
        if len(self.weights.shape) == 1:
            row = self.weights.shape[0]
            col = 1
        else:
            row, col = self.weights.shape
        if self.lesion_mask is None:
            self.lesion_mask = (af.random_uniform(0, 1, size=self.weights.shape) > p).astype(int)

            # self.lesion_mask = af.random.choice([0, 1], size=self.weights.shape, p=[p, 1 - p])
            # self.weights = self.lesion_mask.multiply(self.weights).A if type(self.lesion_mask) is not af.ndarray else self.lesion_mask*(self.weights)

    def heal_by_proportion(self, p):
        if self.lesion_mask is None:
            return
        heal_mask = af.random_uniform(0, 1, size=self.weights.shape) > (1 - p)
        self.lesion_mask[heal_mask] = 1

    def specific_lesion_link(self, links_to_lesion):
        """Lesion p (between 0-1) of connection randomly, if the drop probability is larger than 66%,
        then use the sparse matrix as mask, otherwise use numpy dense matrix as mask."""
        if len(self.weights.shape) == 1:
            row = self.weights.shape[0]
            col = 1
        else:
            row, col = self.weights.shape
        link_specific_lesion_mask = af.ones((row, col)).astype(int)
        if len(links_to_lesion[0]) == 1:
            assert (all(0 <= i <= self.weights.shape[0] - 1 for i in links_to_lesion))
            link_specific_lesion_mask[links_to_lesion] = 0
        else:
            assert (all(0 <= i <= self.weights.shape[0] - 1 and 0 <= j <= self.weights.shape[1] - 1 for i, j in
                        links_to_lesion))
            l, r = [], []
            for i, j in links_to_lesion:
                l.append(i)
                r.append(j)
            link_specific_lesion_mask[l, r] = 0
        if self.lesion_mask is None:
            self.lesion_mask = link_specific_lesion_mask
        else:
            self.lesion_mask *= link_specific_lesion_mask

    def heal(self):
        """Heal lesion connections"""
        self.lesion_mask = None

    def link_specific_heal(self, indices=None):
        """Heal specific link connections"""
        if self.lesion_mask is None:
            return
        if len(indices[0]) == 1:
            self.lesion_mask[indices] = 1
        else:
            l, r = [], []
            for i, j in indices:
                l.append(i)
                r.append(j)
            self.lesion_mask[l, r] = 1

    def reset_weights(self, mean=0., rand_range=1.):
        """
        Resets weights using a specified distribution.

        Args:
            mean (float, optional): Mean of the weight distribution. Defaults to 0.
            rand_range (float, optional): Range of the weight distribution. Defaults to 1.
        """
        if self.initialization == "gaussian":
            weights = af.random_normal(loc=mean, scale=rand_range, size=self.weights.shape)
        elif self.initialization == "kaiming":
            weights = self.kaiming_normal(self.outgoing_group, self.incoming_group, dropout_rate=self.dropout_rate,
                                          perma_lesion_rate=self.perma_lesion_rate).weights
        else:
            weights = af.random_uniform(low=mean-rand_range, high=mean+rand_range, size=self.weights.shape)

        if self.freeze_mask:
            self.weights = (weights * (1 - self.freeze_mask)) + (self.freeze_mask * self.weights)
        else:
            self.weights = weights

    def freeze_incoming_links(self, freeze_all=False, unit_indices=None, link_indices=None):
        """
        Freeze all the incoming links for units at unit_indices or specific (i,j) pairs of links
        """
        row, col = self.weights.shape

        if freeze_all is True:
            self.freeze_mask = af.zeros((row, col))
            return

        freeze_mask = af.ones((row, col)).astype(int)
        if unit_indices:
            print("IN HERE")
            for unit in unit_indices:
                freeze_mask[:, unit] = 0
            self.freeze_mask = freeze_mask
        if link_indices:
            l, r = [], []
            for i, j in link_indices:
                l.append(i)
                r.append(j)
            self.freeze_mask = freeze_mask
            self.freeze_mask[l, r] = 0

    def thaw_incoming_links(self, thaw_all=False, unit_indices=None, link_indices=None):
        """
        Thaw all the incoming links for units at indices
        """
        row, col = self.weights.shape
        if thaw_all is True:
            self.freeze_mask = None

        if self.freeze_mask is not None:
            if unit_indices:
                for unit in unit_indices:
                    self.freeze_mask[:, unit] = 1
            if link_indices:
                l, r = [], []
                for i, j in link_indices:
                    l.append(i)
                    r.append(j)
                self.freeze_mask[l, r] = 1
            if (self.freeze_mask == af.ones((row, col))).all():
                self.freeze_mask = None

    def connection_drop_out(self, p: float):
        """drop p% of the connections randomly at each forward pass. If the drop probability is larger than 66%,
        then use the sparse matrix as mask, otherwise use numpy dense matrix as mask.

        :param p: drop probability
        :type p: float
        """
        if p is None or p <= 0:
            return
        self.dropout_rate = p
        self.dropout_mask = (af.random_uniform(0, 1, size=self.weights.shape) > p).astype(int)

        # the output result will have 2 dimension no matter if the original weight has only 1 dimension
        # self.weights = self.dropout_mask.multiply(self.weights).A if type(
        #     self.dropout_mask) is not af.ndarray else self.dropout_mask * (self.weights)

    def add_noise(self, noise_type, p, z):
        """
        Add noise that are at most z standard deviation away from the mean of the weight entries.

        :param z: z score of the desired range
        :type z: float
        :param noise_type: distribution
        :type noise_type: str
        """
        if p is not None:
            self.noise_mask = (af.random_uniform(0, 1, size=self.weights.shape) > p).astype(int)
        else:
            self.noise_mask = af.ones(self.weights.shape)
        self.pre_noise_weight = self.weights
        deviation = self.weights.std() * z
        if noise_type == "uniform":
            n = af.random_uniform(-deviation, deviation, self.weights.shape)
        elif noise_type == "gaussian":
            n = af.random_normal(-deviation, deviation, self.weights.shape)
        elif noise_type == "kaiming":
            n = af.random_normal(-1, 1, self.weights.shape) * math.sqrt(
                2 / self.weights.shape[0])
        else:
            n = 0
        self.weights += (n * self.noise_mask)

    def remove_noise(self):
        self.weights = (self.pre_noise_weight * self.noise_mask) + (self.weights * (1 - self.noise_mask))

    def forward(self, output_unit):
        """Compute dot product between link.weight and output_unit of the outgoing layer. This will be multiplied by lesion mask if any links are lesioned.

        :param output_unit: the output of layer linking to it
        :type output_unit: af.array
        :return: af.array
        """
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.connection_drop_out(self.dropout_rate)
        else:
            self.dropout_mask = None
        result_weight = self.weights

        if self.dropout_mask is not None:
            result_weight = self.dropout_mask.multiply(result_weight).A if type(
                self.dropout_mask) is not af.ndarray else self.dropout_mask * result_weight
        if self.lesion_mask is not None:
            result_weight = self.lesion_mask.multiply(result_weight).A if type(
                self.lesion_mask) is not af.ndarray else self.lesion_mask * result_weight
        if self.connection_mask is not None:
            result_weight = self.connection_mask.multiply(result_weight).A if type(
                self.connection_mask) is not af.ndarray else self.connection_mask * result_weight

        result = af.dot(output_unit, result_weight)
        return result

    def backward(self, input_derivs: object):
        """
        Compute backward base on the input_unit’s weight derivative. This will be multiplied by the lesion mask if any links are lesioned.

        :param input_derivs: the derivative that this weight link to
        :type input_derivs: af.array
        """

        if self.freeze_mask is not None:
            self.weight_derivs += af.multiply(self.freeze_mask, self.outgoing_group.output_matrix[:, None] @ input_derivs[None, :])
        else:
            if self.outgoing_group.network.parallel_mode:
                self.weight_derivs = self.weight_derivs + self.outgoing_group.output_matrix[:, None] @ input_derivs[None, :]
            else:
                 self.weight_derivs += self.outgoing_group.output_matrix[:, None] @ input_derivs[None, :]
        if self.lesion_mask is not None:
            self.weight_derivs = self.lesion_mask.multiply(self.weight_derivs).A if type(
                self.lesion_mask) is not af.ndarray else self.lesion_mask * self.weight_derivs
        if self.dropout_mask is not None:
            self.weight_derivs = self.dropout_mask.multiply(self.weight_derivs).A if type(
                self.dropout_mask) is not af.ndarray else self.dropout_mask * self.weight_derivs
        if self.connection_mask is not None:
            self.weight_derivs = self.connection_mask.multiply(self.weight_derivs).A if type(
                self.connection_mask) is not af.ndarray else self.connection_mask * self.weight_derivs
        # print("incoming group: {}".format(self.incoming_group.name))
        # print("WEIGHT_DERIVS:")
        # print(self.weight_derivs)
        # print("-------------------------------")

    def backward_prod(self, input_derivs: object, v: float):
        """
        Compute backward base on the input_unit’s weight derivative. This will be multiplied by the lesion mask if any links are lesioned.

        :param input_derivs: the derivative that this weight link to
        :type input_derivs: af.array
        """

        if not self.frozen:
            self.weight_derivs += self.outgoing_group.output_matrix * v

            # print("weight derivs after: {}".format(self.weight_derivs))

            if self.lesion_mask is not None:
                self.weight_derivs = self.lesion_mask.multiply(self.weight_derivs).A if type(
                    self.lesion_mask) is not af.ndarray else self.lesion_mask * self.weight_derivs
            if self.dropout_mask is not None:
                self.weight_derivs = self.dropout_mask.multiply(self.weight_derivs).A if type(
                    self.dropout_mask) is not af.ndarray else self.dropout_mask * self.weight_derivs
            if self.connection_mask is not None:
                self.weight_derivs = self.connection_mask.multiply(self.weight_derivs).A if type(
                    self.connection_mask) is not af.ndarray else self.connection_mask * self.weight_derivs
        # print("outgoing group: {}".format(self.outgoing_group.name))
        # print("incoming group: {}".format(self.incoming_group.name))
        # print("WEIGHT_DERIVS:")
        # print(self.weight_derivs)
        # print("-------------------------------")

    def update_weight(self, new_weight):
        """
        Update the weight base on given weight depending on whether the links are frozen or not.

        :param new_weight: weight that this link need to be updated with
        :type new_weight: af.array
        """
        if not self.frozen:
            self.weights = new_weight

    @classmethod
    def uniform(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None,
                perma_lesion_rate=None):
        """
        Initialize weights from the uniform random distribution

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
        raise NotImplementedError
        # weights = af.random.uniform(mean - range, mean + range, (outgoing_group.num_units, incoming_group.num_units))
        # return cls(outgoing_group, incoming_group, weights)

    @classmethod
    def gaussian(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None,
                 perma_lesion_rate=None):
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
        raise NotImplementedError
        # weights = af.random.normal(mean - range, mean + range, (outgoing_group.num_units, incoming_group.num_units))
        # return cls(outgoing_group, incoming_group, weights)

    @classmethod
    def kaiming_normal(cls, outgoing_group, incoming_group, dropout_rate=None,
                       perma_lesion_rate=None):
        """
        Initialize weights from the kaiming normal distribution

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
        raise NotImplementedError
        # weights = af.random.normal(-1, 1, (outgoing_group.num_units, incoming_group.num_units)) * math.sqrt(
        #     2 / outgoing_group.num_units)
        # return cls(outgoing_group, incoming_group, weights)

    def reset_weight_derivs(self):
        """
        Resets the weight derivative

        Executed after a weight update has occurred
        """
        self.weight_derivs = af.zeros(self.weight_derivs.shape)

    def to_json(self):
        result = {}
        for name, data in self.__dict__.items():
            if name == 'incoming_group' or name == 'outgoing_group':
                # result[name] = data
                result[name] = data.name
            else:
                result[name] = data
        return result

    def from_json(self, outgoing_group, incoming_group, data):
        for key in data:
            setattr(self, key, data[key])

        self.weights = af.array(self.weights)
        self.weight_derivs = af.array(self.weight_derivs)
        self.last_weight_delta = af.array(self.last_weight_delta)
        self.outgoing_group = outgoing_group
        self.incoming_group = incoming_group
        # if self.lesion_rate:
        #     print("non zero lesion rate", outgoing_group.name, incoming_group.name)
        #     if self.lesion_rate > 0.66 and len(self.weights.shape) != 1:
        #         self.lesion_mask = csc_matrix(self.lesion_mask)
        #     else:
        #         self.lesion_mask = af.array(self.lesion_mask)

        if self.dropout_rate:
            if self.dropout_rate > 0.66 and len(self.weights.shape) != 1:
                self.dropout_mask = csc_matrix(self.dropout_mask)
            else:
                self.dropout_mask = af.array(self.dropout_mask)

    def __eq__(self, other):
        """Check if the two link object are equal
        Links are equal if their weight size and projection type are equal and are connecting the same set of groups."""
        if other is None:
            return False
        shape_eq = other.weights.shape == self.weights.shape
        proj_type_eq = other.proj_type == self.proj_type
        same_output_group = self.outgoing_group.name == other.outgoing_group.name
        same_input_group = self.incoming_group.name == other.incoming_group.name
        return shape_eq and proj_type_eq and same_output_group and same_input_group
