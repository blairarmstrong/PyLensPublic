from .link import Link
import math
from ..parameters import LinkParameters
from ..array_factory import Array_factory as af


class LinkFull(Link):
    """Full connected links (weights) between groups (layers) of neural network. (operation: dot product)
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
    # :param link_learning_rate: Link level learning rate, this will override network level learning rate
    # :type link_learning_rate: float
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
    connection_mask: object
    frozen: bool
    link_learning_rate: float
    last_weight_delta: object
    max_weights: float
    min_weights: float
    initialization: str
    link_type: str | None

    def __init__(self, outgoing_group, incoming_group, weights, dropout_rate=None, perma_lesion_rate=None, initialization=None, link_type=None):
        super().__init__(outgoing_group, incoming_group, weights, dropout_rate=dropout_rate,
                         perma_lesion_rate=perma_lesion_rate, initialization=initialization, link_type=link_type)
        self.link_params = LinkParameters()
        self.link_learning_rate = self.link_params.PAR_L_learning_rate
        self.proj_type = 'full'

    @classmethod
    def uniform(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None, perma_lesion_rate=None, initialization=None, link_type=None):
        """
        Initializes weights from a uniform random distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            mean (int): Mean of the uniform distribution.
            range (int): Range/variance of the uniform distribution.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkFull: An instance of LinkFull with uniform-initialized weights.
        """
        weights = af.random_uniform(mean - range, mean + range, (outgoing_group.num_units, incoming_group.num_units))
        return LinkFull(outgoing_group, incoming_group, weights, dropout_rate,
                        perma_lesion_rate, initialization=initialization, link_type=link_type)

    @classmethod
    def gaussian(cls, outgoing_group, incoming_group, mean, range, dropout_rate=None, perma_lesion_rate=None, initialization=None, link_type=None):
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
            LinkFull: An instance of LinkFull with Gaussian-initialized weights.
        """
        weights = af.random_normal(loc=mean, scale=range, size=(outgoing_group.num_units, incoming_group.num_units))
        return LinkFull(outgoing_group, incoming_group, weights, dropout_rate,
                        perma_lesion_rate, initialization=initialization, link_type=link_type)

    @classmethod
    def kaiming_normal(cls, outgoing_group, incoming_group, dropout_rate=None, perma_lesion_rate=None, initialization=None, link_type=None):
        """
        Initializes weights using Kaiming normal distribution.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkFull: An instance of LinkFull with Kaiming-normal initialized weights.
        """
        weights = af.random_normal(-1, 1, (outgoing_group.num_units, incoming_group.num_units)) * math.sqrt(
            2 / outgoing_group.num_units)
        return LinkFull(outgoing_group, incoming_group, weights, dropout_rate,
                        perma_lesion_rate, initialization=initialization, link_type=link_type)

    # TODO: remove
    @classmethod
    def load(cls, outgoing_group, incoming_group, weights, dropout_rate=None, perma_lesion_rate=None):
        """
        Loads pre-trained weights into a LinkFull instance.

        Args:
            outgoing_group: The sending group.
            incoming_group: The receiving group.
            weights (af.array): Pre-trained weight matrix.
            dropout_rate (float, optional): Dropout rate. Defaults to None.
            perma_lesion_rate (float, optional): Permanent lesion rate. Defaults to None.
        
        Returns:
            LinkFull: An instance of LinkFull with pre-trained weights.
        """
        return LinkFull(outgoing_group, incoming_group, weights, dropout_rate, perma_lesion_rate)

    def reset_weight_derivs(self):
        """
        Resets the weight derivatives to zero after a weight update has occurred.
        """
        self.weight_derivs = af.zeros(self.weight_derivs.shape)
