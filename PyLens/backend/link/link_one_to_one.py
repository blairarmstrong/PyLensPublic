from .link import Link
import math
# from src.backend.parameters import LinkParameters
from ..parameters import LinkParameters
from ..array_factory import Array_factory as af


# from src.backend.defaults.defaults import LinkDefaults


class LinkOneToOne(Link):
    """One to one connected links (weights) between groups (layers) of neural network.  (operation: hadamard product)
    """

    weights: object
    weight_derivs: object
    outgoing_group: "Group"
    incoming_group: "Group"
    lesion_mask: object
    connection_mask: object
    links_to_lesion = None
    frozen: bool
    alpha: float
    last_weight_delta: object
    max_weights: float
    min_weights: float
    link_type: str

    def __init__(self, outgoing_group, incoming_group, weights, dropout_rate=None,
                 perma_lesion_rate=None, link_type=None):
        super().__init__(outgoing_group, incoming_group, weights, dropout_rate=dropout_rate,
                         perma_lesion_rate=perma_lesion_rate, link_type=link_type)
        link_parameters = LinkParameters()
        self.link_learning_rate = link_parameters.PAR_L_learning_rate
        self.proj_type = 'one-to-one'
        self.link_type = link_type
        # the two layer must have same number of hidden units for one to one to be possible
        assert outgoing_group.num_units == incoming_group.num_units, 'Number of hidden unit must match for one to one connection'

    def forward(self, output_unit):
        """
        Computes the Hadamard product between link weights and the outgoing layer's output.

        Args:
            output_unit (af.array): Output of the layer sending the link.
        
        Returns:
            af.array: The element-wise product of output_unit and link weights.
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
        print("------ link one to one self weight: ", self.weights)
        print("------ link one to one output_unit: ", self.weights)

        # return self.weights
        return af.multiply(output_unit, result_weight)

    def backward(self, input_derivs):
        """
        Computes the gradient update during backpropagation.

        Args:
            input_derivs (af.array): Gradient from the next layer.
        """

        if not self.frozen:
            if self.freeze_mask:
                self.weight_derivs += self.freeze_mask.diagonal().multiply((
                    self.outgoing_group.output_matrix[:, None] @ input_derivs[None, :]).diagonal())
            else:
                self.weight_derivs += (
                    self.outgoing_group.output_matrix[:, None] @ input_derivs[None, :]).diagonal()
            if self.lesion_mask:
                self.weight_derivs = self.lesion_mask.multiply(self.weight_derivs).A if type(
                    self.lesion_mask) is not af.ndarray else self.lesion_mask * self.weight_derivs
            # self.weight_derivs *= af.identity(self.weight_derivs.shape[0])
            if self.dropout_mask is not None:
                self.weight_derivs = self.dropout_mask.multiply(self.weight_derivs).A if type(
                    self.dropout_mask) is not af.ndarray else self.dropout_mask * self.weight_derivs
            if self.connection_mask is not None:
                self.weight_derivs = self.connection_mask.multiply(self.weight_derivs).A if type(
                    self.connection_mask) is not af.ndarray else self.connection_mask * self.weight_derivs

    @classmethod
    def uniform(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None,
                perma_lesion_rate=None, link_type=None):
        """
        Initializes 1D weights from a uniform random distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            mean (int): Mean of the uniform distribution.
            range (int): Range of the uniform distribution.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkOneToOne: An instance of LinkOneToOne with uniform-initialized weights.
        """
        # print(incoming_group)
        weights = af.random_uniform(mean - range, mean + range, incoming_group.num_units)
        # weights = af.diag(weights)
        # weights = diags(weights, 0)
        link = LinkOneToOne(outgoing_group, incoming_group, weights, dropout_rate,
                            perma_lesion_rate, link_type=link_type)
        # weights are frozen by default for 1-to-1
        link.freeze()
        return link

    @classmethod
    def gaussian(cls, outgoing_group, incoming_group, mean, range,
                 dropout_rate=None,
                 perma_lesion_rate=None, link_type=None):
        """
        Initializes 1D weights from a Gaussian random distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            mean (int): Mean of the Gaussian distribution.
            range (int): Standard deviation of the Gaussian distribution.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkOneToOne: An instance of LinkOneToOne with Gaussian-initialized weights.
        """
        weights = af.random_normal(mean, range, incoming_group.num_units)
        # weights = af.diag(weights)
        # weights = diags(weights, 0)
        link = LinkOneToOne(outgoing_group, incoming_group, weights, dropout_rate,
                            perma_lesion_rate, link_type=link_type)
        # weights are frozen by default for 1-to-1
        link.freeze()
        return link

    @classmethod
    def kaiming_normal(cls, outgoing_group, incoming_group,  dropout_rate=None,
                       perma_lesion_rate=None, link_type=None):
        """
        Initializes 1D weights using Kaiming normal distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkOneToOne: An instance of LinkOneToOne with Kaiming-normal initialized weights.
        """
        weights = af.random_normal(-1, 1, (incoming_group.num_units)) * math.sqrt(
            2 / outgoing_group.num_units)
        # weights = af.diag(weights)
        # weights = diags(weights, 0)
        link = LinkOneToOne(outgoing_group, incoming_group, weights, dropout_rate,
                            perma_lesion_rate, link_type=link_type)
        # weights are frozen by default for 1-to-1
        link.freeze()
        return link

    # TDOO remove
    @classmethod
    def load(cls, outgoing_group, incoming_group, weights, dropout_rate=None,
             perma_lesion_rate=None):
        """
        Initialize weights from the parameter

        :param cls: this is equivalent to self, creating a instantiation of this Link object
        :type cls: Link
        :param outgoing_group: the sending group
        :type outgoing_group: Group
        :param incoming_group: the receiving group
        :type incoming_group: Group
        :param weights: the pretrained weight
        :type mean: numpy.array
        """
        return LinkOneToOne(outgoing_group, incoming_group, weights, dropout_rate,
                            perma_lesion_rate)

    def reset_weight_derivs(self):
        """
        Resets the weight derivative

        Executed after a weight update has occurred
        """
        self.weight_derivs = af.zeros(self.weights.shape)
