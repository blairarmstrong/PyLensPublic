from .link import Link
import math
from ..array_factory import Array_factory as af
from scipy.sparse import rand
from PyLens.backend.parameters import LinkParameters


# from src.backend.defaults.defaults import LinkDefaults


class LinkRandom(Link):
    """Initialized to drop drop_prob% of link permanently, if the drop probability is larger than 66%,
    then use the sparse matrix as random mask, otherwise use numpy dense matrix as mask.

    """


    # Programmer's guide on sparse vs dense mask choice:
    # Dot product
    # For dot product between n by n weight matrix by a size n vector.
    # Sparse matrix has a timing advantage when over 75% of the cells are unfiled.
    # The time performance: Scipy is y times faster than numpy can be modelled by y = -1/(4(p-1)) as n increases, where p is the drop rate.

    # Elementwise matrix multiplication
    # For dot product between n by n weight matrix by a size n by n matrix.
    # Sparse matrix has a timing advantage when over 60% of the cells are unfiled.
    # The time performance: Scipy is y times faster than numpy can be modelled by y = -3/(8(p-1)) as n increases, where p is the drop rate.

    # =================================================================================

    # :param weights: Weights of the specific link
    # :type weights: af.array
    # :param weight_derivs: Keep track of the gradient of the weights at current backward pass
    # :type weight_derivs: af.array
    # :param outgoing_group: The group (layer) that send this link out
    # :type outgoing_group: Group
    # :param incoming_group: The group (layer) that receives the link
    # :type incoming_group: Group
    # :param lesion_mask: The sparse mask/numpy array of 0 and 1 such that 0 means masking specific value.
    # :type lesion_mask: af.array
    # :param frozen: True if the link’s weight cannot be updated, False if link’s weight can be updated during backpropagation.
    # :type frozen: bool
    # :param link_learning_rate: Link level learning rate, this will override network level learning rate
    # :type link_learning_rate: float
    # :param last_weight_delta: Keep track of the gradient of the weights of previous backward pass
    # :type last_weight_delta: af.array
    # :param max_weights: An upper bound on the values in weight matrix
    # :type max_weights: float
    # :param min_weights: An lower bound on values in weight matrix
    # :type min_weights: float
    # :param random_mask: The sparse mask/numpy array of 0 and 1 such that 0 means masking specific value.
    # :type random_mask: str

    weights: object
    weight_derivs: object
    outgoing_group: "Group"
    incoming_group: "Group"
    lesion_mask: object
    connection_mask: object
    frozen: bool
    link_learning_rate: float
    last_weight_delta: object
    max_weights: float
    min_weights: float
    initialization: str
    link_type: str |None
    random_mask: object

    def __init__(self, outgoing_group, incoming_group, weights, dropout_rate=None, perma_lesion_rate=None, initialization=None, link_type=None):
        super().__init__(outgoing_group, incoming_group, weights, dropout_rate=dropout_rate,
                         perma_lesion_rate=perma_lesion_rate, initialization=initialization, link_type=link_type)
        link_default = LinkParameters()
        self.link_learning_rate = link_default.PAR_L_learning_rate
        self.perma_lesion_rate = perma_lesion_rate
        self.initialization=initialization
        self.proj_type = 'random'
        if len(self.weights.shape) == 1:
            row = self.weights.shape[0]
            col = 1
        else:
            row, col = self.weights.shape
        if perma_lesion_rate is not None and perma_lesion_rate > 0:
            if perma_lesion_rate > 0.66 and len(self.weights.shape) > 1:
                self.random_mask = rand(row, col, density=1 - perma_lesion_rate, format="csc", random_state=42)
                self.random_mask.data[:] = 1
                # self.weights = self.random_mask.multiply(self.weights).toarray()
            else:
                self.random_mask = af.random_uniform(0, 1, size=self.weights.shape) >= perma_lesion_rate
                # self.weights = self.random_mask * self.weights
        else:
            self.random_mask = None

    def get_type(self):
        return 'random'

    def forward(self, output_unit: object):
        """
        Computes the dot product between link weights and the outgoing layer's output.

        Args:
            output_unit (af.array): Output of the layer sending the link.
        
        Returns:
            af.array: The dot product result.
        """
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.connection_drop_out(self.dropout_rate)
        else:
            self.dropout_mask = None
            self.dropout_rate = None
        result_weight = self.weights
        if self.dropout_mask is not None:
            result_weight = self.dropout_mask.multiply(result_weight).A if type(
                self.dropout_mask) is not af.ndarray else self.dropout_mask * result_weight
        if self.lesion_mask is not None:
            result_weight = self.lesion_mask.multiply(result_weight).A if type(
                self.lesion_mask) is not af.ndarray else self.lesion_mask * result_weight
        if self.random_mask is not None:
            result_weight = self.random_mask.multiply(result_weight).A if type(
                self.random_mask) is not af.ndarray else self.random_mask * result_weight
        if self.connection_mask is not None:
            result_weight = self.connection_mask.multiply(result_weight).A if type(
                self.connection_mask) is not af.ndarray else self.connection_mask * result_weight

        result = af.dot(output_unit, result_weight)
        return result

    def backward(self, input_derivs):
        """
        Computes the gradient update during backpropagation.

        Args:
            input_derivs (af.array): Gradient from the next layer.
        """
        # print('backward dropout',self.dropout_mask)
        if not self.frozen:
            self.weight_derivs += (self.outgoing_group.output_matrix[:, None] @ input_derivs[None, :])
            if self.lesion_mask is not None:
                self.weight_derivs = self.lesion_mask.multiply(self.weight_derivs).A if type(
                    self.lesion_mask) is not af.ndarray else self.lesion_mask * self.weight_derivs
            if self.random_mask is not None:
                self.weight_derivs = self.random_mask.multiply(self.weight_derivs).A if type(
                    self.random_mask) is not af.ndarray else self.random_mask * self.weight_derivs
            if self.dropout_mask is not None:
                self.weight_derivs = self.dropout_mask.multiply(self.weight_derivs).A if type(
                    self.dropout_mask) is not af.ndarray else self.dropout_mask * self.weight_derivs
            if self.connection_mask is not None:
                self.weight_derivs = self.connection_mask.multiply(self.weight_derivs).A if type(
                    self.connection_mask) is not af.ndarray else self.connection_mask * self.weight_derivs

    @classmethod
    def uniform(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None, perma_lesion_rate=None, initialization=None, link_type=None):
        """
        Initializes weights from a uniform random distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            mean (int): Mean of the uniform distribution.
            range (int): Range of the uniform distribution.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkRandom: An instance of LinkRandom with uniform-initialized weights.
        """
        weights = af.random_uniform(mean - range, mean + range, (outgoing_group.num_units, incoming_group.num_units))
        return LinkRandom(outgoing_group, incoming_group, weights, dropout_rate,
                          perma_lesion_rate, initialization=initialization, link_type=link_type)

    @classmethod
    def gaussian(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None,
                 perma_lesion_rate=None, initialization=None, link_type=None):
        """
        Initializes weights from a Gaussian random distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            mean (int): Mean of the Gaussian distribution.
            range (int): Standard deviation of the Gaussian distribution.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkRandom: An instance of LinkRandom with Gaussian-initialized weights.
        """
        weights = af.random_normal(loc=mean, scale=range, size=(outgoing_group.num_units, incoming_group.num_units))
        return LinkRandom(outgoing_group, incoming_group, weights, dropout_rate, perma_lesion_rate, initialization=initialization, link_type=link_type)

    @classmethod
    def kaiming_normal(cls, outgoing_group, incoming_group, dropout_rate=None,
                       perma_lesion_rate=None, initialization=None, link_type=None):
        """
        Initializes weights using Kaiming normal distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkRandom: An instance of LinkRandom with Kaiming-normal initialized weights.
        """
        weights = af.random_normal(-1, 1, (outgoing_group.num_units, incoming_group.num_units)) * math.sqrt(
            2 / outgoing_group.num_units)
        return LinkRandom(outgoing_group, incoming_group, weights, dropout_rate,
                          perma_lesion_rate, initialization=initialization, link_type=link_type)

    @classmethod
    def load(cls, outgoing_group, incoming_group, weights, dropout_rate=None, perma_lesion_rate=None):
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
        return LinkRandom(outgoing_group, incoming_group, weights, dropout_rate, perma_lesion_rate)

    def reset_weight_derivs(self):
        """
        Resets the weight derivatives to zero after a weight update.
        """
        self.weight_derivs = af.zeros(self.weights.shape)
