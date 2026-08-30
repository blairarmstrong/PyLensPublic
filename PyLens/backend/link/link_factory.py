from .link_full import LinkFull
from .link_random import LinkRandom
from .link_one_to_one import LinkOneToOne
from ..array_factory import Array_factory as af
from scipy.sparse.csc import csc_matrix


class LinkFactory:
    @staticmethod
    def store_link(link):
        """
        Serializes a given link into a JSON-compatible format.

        Args:
            link: The link object to be serialized.
        
        Returns:
            dict: A JSON-compatible dictionary representation of the link.
        """
        return link.to_json()

    @staticmethod
    def load_into_new_link(outgoing_group, incoming_group, data):
        """
        Loads link data into a newly constructed link object.

        Args:
            outgoing_group: The outgoing group of the link.
            incoming_group: The incoming group of the link.
            data (dict): JSON data containing the projection type and other parameters.
        
        Returns:
            Link: A newly created link object with loaded data.
        """
        target_proj_type = data['proj_type']
        link = LinkFactory.construct_link(outgoing_group, incoming_group, 'uniform', rand_mean=0, rand_range=1,
                                          proj_type=target_proj_type, dropout_rate=0,
                                          perma_lesion_rate=0)
        link.from_json(outgoing_group, incoming_group, data)
        return link

    @staticmethod
    def load_into_given_link(link, outgoing_group, incoming_group, data):
        """
        Loads link data into an existing link object.
        """
        link.from_json(outgoing_group, incoming_group, data)

    @staticmethod
    def construct_link(
            outgoing_group, 
            incoming_group, 
            initialization, 
            rand_mean=0, 
            rand_range=1, 
            proj_type="full", 
            link_type=None,
            dropout_rate=None, 
            perma_lesion_rate=None
            ):
        """
        Constructs a new link based on the specified type and parameters.

        Args:
            outgoing_group
            incoming_group
            initialization (str): The random initialization of link ('uniform', 'gaussian', 'kaiming').
            rand_mean (float, optional): Mean value for random initialization. Defaults to 0.
            rand_range (float, optional): Range value for random initialization. Defaults to 1.
            proj_type (str, optional): The projection type ('full', 'one-to-one', 'random'). Defaults to "full".
            dropout_rate (float, optional): The dropout rate for the link. Defaults to None.
            perma_lesion_rate (float, optional): The permanent lesion rate for the link. Defaults to None.
        
        Returns:
            Link: A newly constructed link object of the specified type.
        """
        proj_types = {"full": LinkFull, 'one-to-one': LinkOneToOne, 'random': LinkRandom}
        link_class = proj_types[proj_type]
        if initialization == "uniform":
            return link_class.uniform(outgoing_group, incoming_group, rand_mean, rand_range, dropout_rate=dropout_rate,
                                      perma_lesion_rate=perma_lesion_rate, initialization=initialization, link_type=link_type)
        elif initialization == "gaussian":
            return link_class.gaussian(outgoing_group, incoming_group, rand_mean, rand_range, dropout_rate=dropout_rate,
                                       perma_lesion_rate=perma_lesion_rate, initialization=initialization, link_type=link_type)
        elif initialization == "kaiming":
            return link_class.kaiming_normal(outgoing_group, incoming_group, dropout_rate=dropout_rate,
                                             perma_lesion_rate=perma_lesion_rate, initialization=initialization, link_type=link_type)
        else:
            raise ValueError(
                f"Unknown initialization method: {initialization}"
            )

    @staticmethod
    def store_data_type_converter(obj):
        """
        Converts data types that are not JSON serializable into a compatible format.

        Args:
            obj: The object to be converted (can be a numpy array or sparse matrix).
        
        Returns:
            list: A JSON-compatible list representation of the object.
        
        Raises:
            TypeError: If the object type is not supported for conversion.
        """
        if isinstance(obj, af.ndarray):
            return obj.tolist()
        if isinstance(obj, csc_matrix):
            return obj.A.tolist()
        raise TypeError('Not serializable')
