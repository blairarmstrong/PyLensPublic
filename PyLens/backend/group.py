import os
import sys
import copy

from .array_factory import Array_factory as af

## Parameters
from .parameters import GroupParameters

## Input Transformations
from .inputs.soft_clamp import Soft_Clamp
from .inputs.in_copy import In_Copy
from .inputs.in_integr import In_Integr
from .inputs.dot_product import Dot_Product
from .inputs.product import Product
from .inputs.distance import Distance
from .inputs.boltzmann import BoltzmannInput

## Output Transformations
from .output_transforms.cropped import Cropped
from .output_transforms.elman_clamp import Elman_Clamp
from .output_transforms.exponential import Exponential
from .output_transforms.gaussian import Gaussian
from .output_transforms.hard_clamp import Hard_Clamp
from .output_transforms.linear import Linear
from .output_transforms.noise import Noise
from .output_transforms.out_integr import Out_Integr
from .output_transforms.sigmoid import Sigmoid
from .output_transforms.soft_max import SoftMax
from .output_transforms.tanh import Tanh
from .output_transforms.ternary import Ternary
from .output_transforms.out_copy import Out_Copy
from .output_transforms.bias_clamp import Bias_Clamp
from .output_transforms.weak_clamp import Weak_Clamp
from .output_transforms.out_winner import Out_Winner
from .output_transforms.out_deriv_noise import Out_Deriv_Noise
from .output_transforms.out_norm import Out_Norm
from .output_transforms.interact_integr import Interact_Integr
from .output_transforms.kohonen import Kohonen
from .output_transforms.boltzmann import BoltzmannOutput

## Links
from .link.link_full import LinkFull
from .link.link_one_to_one import LinkOneToOne
from .link.link_random import LinkRandom

group_params = GroupParameters()

basic_output_transforms = (
    Linear,
    Sigmoid,
    Ternary,
    Tanh,
    Gaussian,
    Exponential,
    SoftMax,
    Kohonen,
    BoltzmannOutput,
    Out_Copy,
    Interact_Integr,
    Bias_Clamp,
    # Hard_Clamp
)


class GroupProc:
    """
    A helper class for managing group processing data.

    This class is responsible for storing historical and intermediate 
    computation data for a `Group` during simulation.

    """
    def __init__(self, name, group):
        self.name = name
        self.group = group
        self.unitHistoryData = af.empty((self.group.max_ticks, group.num_units))
        self.unitData = af.zeros(group.num_units)
        self.groupHistoryData = af.empty((self.group.max_ticks))


class Group:
    """
    This class represents a group (also called layer) of units.  It is responsible for processing inputs, applying input transformations, and computing output activations.

    """
    # links will hold previous input data to initialize weight matrices
    link = None
    incoming_links = None
    outgoing_links = None

    # initialize basic properties of all groups
    name = None
    num_units = 0
    group_type = 0
    external_input = None
    target = None
    input_matrix = None
    output_matrix = None
    output_matrix_cache = None
    input_derivs = None
    output_derivs = None
    incoming_derivs = None
    incoming_weights = None
    input_history = None
    output_history = None
    output_derivs_history = None
    bias = None
    input_transforms = []
    output_transforms = []
    input_set = False
    num_cols = None
    curr_tick = 0
    names = []
    unit_names = []
    total_deriv = None
    weight_elimination = None
    error_scale = group_params.PAR_G_errorScale

    # lookup table matching different transforms to their respective classes
    # activations = {"sigmoid": Sigmoid, "hard_clamp": Hard_Clamp, "soft_clamp": Soft_Clamp, "linear": Linear, "soft_max": SoftMax, "noise": Noise, "cropped" : Cropped,
    #                "gaussian": Gaussian, "out_integr": Out_Integr}
    activations = {"sigmoid": Sigmoid, "hard_clamp": Hard_Clamp,
                   "linear": Linear, "soft_max": SoftMax, "noise": Noise,
                   "cropped": Cropped, "gaussian": Gaussian,
                   "out_integr": Out_Integr, "elman_clamp": Elman_Clamp,
                   "exponential": Exponential, "tanh": Tanh, "ternary": Ternary,
                   "out_copy": Out_Copy, "bias_clamp": Bias_Clamp,
                   "weak_clamp": Weak_Clamp, "out_winner": Out_Winner,
                   "out_deriv_noise": Out_Deriv_Noise, "out_norm": Out_Norm,
                   "interact_integr": Interact_Integr, "kohonen": Kohonen,
                   "boltzmann": BoltzmannOutput}
    # No Backward Activation Function
    no_back_out = {"Out_Copy", "Interact_integr"}
    # input_types = {"dot": Dot_Product, "product": Product, "elman_clamp": Elman_Clamp, "in_integr": In_Integr}
    input_types = {"dot": Dot_Product, "product": Product,
                   "soft_clamp": Soft_Clamp,"in_copy": In_Copy,
                   "in_integr": In_Integr, "distance": Distance,
                   "boltzmann": BoltzmannInput}
    total_group_types = ["input", "hidden", "output", "bias", "elman"]
    proj_type = {"full": LinkFull, 'one-to-one': LinkOneToOne, 'random': LinkRandom}

    def __init__(self, name, num_units, group_type, input_transforms, output_transforms, time_intervals,
                 ticks_per_interval, network, dropout_rate=None, num_cols=None):
        self.network = network
        self.max_ticks = self.network.max_ticks
        self.time_intervals = time_intervals
        self.ticks_per_interval = ticks_per_interval
        self.gain = group_params.PAR_G_gain
        self.noise_proc = group_params.PAR_G_noiseProc
        self.noise_range = group_params.PAR_G_noiseRange
        self.clamp_strength = group_params.PAR_G_clampStrength
        self.output_cost_scale = group_params.PAR_G_outputCostScale
        self.output_cost_peak = group_params.PAR_G_outputCostPeak
        # Kohonen
        self.neighborhood = group_params.PAR_G_neighborhood
        self.periodicBoundary = group_params.PAR_G_periodicBoundary
        self.ternary_shift = group_params.PAR_G_ternaryShift
        self.name = name
        self.dt = 1.
        self.baseType = os.getenv('BASETYPE')
        # append 1 to number of units to account for bias unit
        self.num_units = num_units
        [self.minOutput, self.maxOutput, self.initOutput] = group_params.get_min_max_initOut(self)
        self.initInput = group_params.PAR_G_initInput

        self.group_type = group_type

        self.num_cols = num_cols
        self.unit_names = []
        self.lesion_mask = None
        self.dropout_rate = dropout_rate
        self.unit_dropout(self.dropout_rate)

        self.names = [name + str(i) for i in range(0, num_units)]

        # initialize arrays
        self.input_matrix = af.zeros(self.num_units)
        self.output_matrix = af.zeros(self.num_units)
        self.input_derivs = af.zeros(self.num_units)
        self.output_derivs = af.zeros(self.num_units)
        self.outputderivCache = af.zeros(self.num_units)
        # initialize external input and target to NaN for individual unit checking
        self.external_input = af.empty(self.num_units)
        self.external_input[:] = af.NaN
        self.target = af.empty(self.num_units)
        self.target[:] = af.NaN
        self.polarity_sum = 0.0
        self.polarity_num = 0

        # should this be intialized to nan, makes things easier if it is 0
        self.output_derivs_history = af.zeros((self.max_ticks, self.num_units))

        if group_type != "bias":
            self.external_input_history = []
            self.input_history = af.empty((self.max_ticks, self.num_units))
            self.output_history = af.empty((self.max_ticks, self.num_units))
            self.target_history = af.empty((self.max_ticks, self.num_units))
            self.output_history[:] = af.NaN
            self.input_history[:] = af.NaN
            self.target_history[:] = af.NaN
        else:
            self.output_history = af.array([1])
            self.input_history = af.array([0.])
            self.target_history = af.array([af.NaN])

        self.incoming_links = []
        self.outgoing_links = []
        self.incoming_derivs = af.zeros(self.num_units)
        self.incoming_weights = af.array([[1]])
        self.weight_elimination = None

        self.input_transforms = []
        for transform in input_transforms:
            self.input_transforms.append(self.input_types[transform](self))
        self.output_transforms = []
        for transform in output_transforms:
            self.output_transforms.append(self.activations[transform](self))

        # if the group is an input group, it doesn't apply any transforms (i.e. what you input, will be outputted)
        if self.group_type == "input":
            self.output_transforms.append(self.activations["hard_clamp"](self))
        elif self.group_type == "elman":
            self.output_transforms = [self.activations["elman_clamp"](self)]
            af.fill(self.output_matrix, 0.5)
            self.reset_input = True

    def reset_unit_values(self, field, value=None):
        
        """
        Reset values of a specified field for all units in the group.
        """
        
        field_mapping = {
        "input_matrix": self.input_matrix,
        "output_matrix": self.output_matrix,
        "input_derivs": self.input_derivs,
        "output_derivs": self.output_derivs, 
        "target": self.target,
        "external_input": self.external_input
    }
        # Validate field input
        if field not in field_mapping:
            raise ValueError(f"Invalid field '{field}'. Choose from {list(field_mapping.keys())}")

        # Get the reference to the correct attribute
        attr = field_mapping[field]

        # Set default values if value is None
        if value is None:
            if field in ["target", "external_input"]:
                attr[:] = af.NaN 
            else:
                attr[:] = 0.0 
        else:
            attr.fill(value)  # Fill the array with the specified value
        
    def to_json(self):
        """
        Converts the Group object into a JSON-serializable dictionary.

        Returns:
            dict: A dictionary containing the group's attributes.
        """
        result = {}
        for name, data in self.__dict__.items():
            if isinstance(data, int) or isinstance(data, float) or isinstance(data, str) or isinstance(data, bool):
                result[name] = data
        return result

    def from_json(self, data):
        """
        Loads the group’s attributes from a JSON-serializable dictionary.

        Args:
            data (dict): A dictionary containing the group's attributes.
        """
        for key in data:
            setattr(self, key, data[key])

    def lesion_group(self, p):
        """
        Applies a lesion (disabling) to a proportion of units in the group.

        Args:
            p (float): The proportion of units to lesion.
        """
        p_lesion_mask = af.random_uniform(0, 1, size=self.num_units) > p
        if self.lesion_mask is None:
            self.lesion_mask = p_lesion_mask.astype(int)
        else:
            self.lesion_mask *= p_lesion_mask.astype(int)

    def lesion_units(self, units):
        """
        Lesions specific units in the group.

        Args:
            units (list): List of unit indices to lesion.
        """
        unit_specific_lesion_mask = af.ones(self.num_units)
        unit_specific_lesion_mask[units] = 0
        if self.lesion_mask is None:
            self.lesion_mask = unit_specific_lesion_mask
        else:
            self.lesion_mask *= unit_specific_lesion_mask

    def heal(self):
        """
        Heals all lesioned units, restoring them to full functionality.
        """
        self.lesion_mask = None

    def heal_by_proportion(self, p):
        """
        Heals a proportion of previously lesioned units.
        """
        heal_mask = af.random_uniform(0, 1, size=self.num_units) > (1 - p)
        self.lesion_mask[heal_mask] = 1

    def heal_units(self, indices):
        """
        Heals specific lesioned units.

        Args:
            indices (list): List of unit indices to heal.
        """
        if self.lesion_mask is None:
            return
        assert (isinstance(indices, list))
        assert (all(isinstance(i, int) for i in indices))
        assert (all(0 <= i <= self.num_units - 1 for i in indices))

        self.lesion_mask[indices] = 1

    def add_bias(self, bias):
        """
        Adds a bias group to the incoming link list of the current group.

        Args:
            bias (Group): The bias group.
        """
        bias_link = LinkFull.uniform(bias, self, 0, 1)  # INHERIT NETWORK

        self.incoming_links.append(bias_link)
        bias.outgoing_links.append(bias_link)

    def add_unit_names(self, names):
        """
        Assigns names to the individual units.

        Args:
            names (list): List of unit names.
        """
        if len(names) == self.num_units:
            self.unit_names = names

    def link_previous(self, new_link):
        """
        Links the preceding group to this group.

        Args:
            new_link (Link): The link connecting the previous group.
        """
        self.incoming_links.append(new_link)

    def link_next(self, new_link):
        """
        Links this group to the next group.
        """
        self.outgoing_links.append(new_link)

    def set_external_input(self, input_matrix):
        """
        Sets the external input of the group.

        Args:
            input_matrix (ndarray): The external input matrix.
        """
        self.external_input = af.array(input_matrix)
        self.input_set = True

    def previous_target(self, target_matrix):
        """
        Sets the target matrix.

        Args:
            target_matrix (ndarray): The target values.
        """
        self.target = af.array(target_matrix)

    def compute_input(self):
        """
        Computes the input transformations and updates the input matrix.
        """
        # compute input if group has incoming links
        if self.incoming_links:
            for transform in self.input_transforms:
                self.input_matrix = transform.compute(self.incoming_links)

    def compute_input_back(self):
        """
        Computes the backward pass for input derivatives.

        Returns:
            ndarray: The computed input derivatives.
        """
        input_derivs = af.zeros(self.num_units)
        if self.group_type not in ("bias", "elman") and self.incoming_links:
            for transform in reversed(self.input_transforms):
                input_derivs = transform.backward(self.incoming_links, self.input_derivs)

        return input_derivs

    def compute_output(self):
        """
        Applies output transformations to the computed input.
        """

        if not any(
            isinstance(transform, basic_output_transforms)
            for transform in self.output_transforms
        ):
            self.output_matrix[:] = 0.0

        for transform in self.output_transforms:
            if isinstance(transform, basic_output_transforms):
                self.output_matrix = transform.forward(self.input_matrix)
            else:
                self.output_matrix = transform.forward(self.output_matrix)

        self.cache_outputs()

    def compute_output_back(self):
        """
        Computes the backward pass for output derivatives.

        Returns:
            ndarray: The computed output derivatives.
        """
        # outputs = af.zeros(self.num_units)
        # for transform in reversed(self.output_transforms):
        #     transform_class = self.activations[transform](self)
        #     outputs = transform_class.backward(self.input_matrix, self.output_derivs)
        # return outputs
        self.output_derivs += self.outputderivCache
        # Make a copy to fix assignment destination is read-only
        if self.network.parallel_mode:
            self.outputderivCache = self.outputderivCache.copy()
        af.fill(self.outputderivCache, 0)

        outputs = af.zeros(self.num_units)
        for transform in reversed(self.output_transforms):
            if transform.name not in self.no_back_out:
                outputs = transform.backward(self.input_matrix, self.output_derivs)

        return outputs

    def unit_dropout(self, p: float):
        """
        Applies dropout to randomly disable units during training.
        """
        if p is None or p <= 0:
            self.dropout_mask = None
            return
        self.dropout_rate = p
        self.dropout_mask = af.random_uniform(0, 1, size=(self.num_units)) > p


    def forward(self, tick):
        """
        Performs a forward pass through the group.
        """
        self.compute_input()
        self.compute_output()

        if self.lesion_mask is not None:
            self.output_matrix *= self.lesion_mask

        # reinitialize dropout mask
        self.unit_dropout(self.dropout_rate)
        if self.dropout_mask is not None:
            self.output_matrix *= self.dropout_mask

        self.input_set = False

        # Make a copy to fix assignment destination is read-only
        if self.network.parallel_mode:
            self.input_history = self.input_history.copy()
            self.output_history = self.output_history.copy()
        self.input_history[tick] = self.input_matrix
        self.output_history[tick] = self.output_matrix

        return self.output_matrix

    def backward(self):
        """
        Compute the backward pass (i.e. the derivative of the error with respect to everything else)
        """

        input_derivs = self.compute_output_back()
        if self.group_type != "elman":
            self.input_derivs = input_derivs * self.lesion_mask if self.lesion_mask is not None else input_derivs
            self.input_derivs = self.input_derivs * self.dropout_mask if self.dropout_mask is not None else input_derivs

        else:
            pass
        self.compute_input_back()

        return self.output_derivs

    def cache_outputs(self):
        """
        Stores a copy of the current output matrix.
        """
        self.output_matrix_cache = af.copy(self.output_matrix)

    def reset_report_params(self):
        """
        Reset the group-specific report statistics.
        """
        self.lwd_times_derivs = 0.0
        self.squared_lwd = 0.0
        self.squared_derivs = 0.0
        self.squared_weights = 0.0

    def pre_update_report_stat(self, link):
        """
        Updates report statistics before weight update.

        Args:
            link (Link): The link whose attributes are used to update the statistics.
        """
        self.lwd_times_derivs += af.sum(link.last_weight_delta * link.weight_derivs)
        self.squared_lwd += af.sum(link.last_weight_delta ** 2)
        self.squared_derivs += af.sum(link.weight_derivs ** 2)

    def post_update_report_stat(self, link):
        """
        Updates report statistics after weight update.
        """
        if not link.frozen:
            self.squared_weights += af.sum(link.weights ** 2)

    def aggregate_report_stats(self, report_stats):
        """
        Aggregates group-specific statistics into the overall network report.

        Args:
            report_stats (dict): The network-level report statistics dictionary.
        """
        report_stats["lwd_times_derivs"] += self.lwd_times_derivs
        report_stats["squared_lwd"] += self.squared_lwd
        report_stats["squared_derivs"] += self.squared_derivs
        if self.group_type != "elman":
            report_stats["squared_weights"] += self.squared_weights

    def clear_derivs(self):
        """
        Clears all derivative matrices.
        """
        # Make a copyu to fix assignment desitnation is read only
        if self.network.parallel_mode:
            self.input_derivs = self.input_derivs.copy()
            self.output_derivs = self.output_derivs.copy()
            self.incoming_derivs = self.incoming_derivs.copy()
        af.fill(self.input_derivs, 0)
        af.fill(self.output_derivs, 0)

        af.fill(self.incoming_derivs, 0)
        self.incoming_weights = af.array([[1]])

    def check_params(self):
        """
        Validates group parameters to ensure correctness.
        """
        if isinstance(self.gain, list) and (len(self.gain) != self.num_units or all(gain <= 0 for gain in self.gain)):
            sys.exit("gain value is not a positive number or wrong of number values for units")
        elif (isinstance(self.gain, int)) and self.gain <= 0:
            sys.exit("gain value is not a positive number")
        elif self.maxOutput < self.minOutput:
            sys.exit("maxOutput is less than minOutput")
        elif self.noise_proc not in {"addGaussianNoise", "addUniformNoise", "multiplyUniformNoise",
                                     "multiplyGaussianNoise"}:
            sys.exit(
                "noise_proc has to be 'addGaussianNoise' or 'addUniformNoise' or 'multiplyGaussianNoise' or 'multiplyUniformNoise'")
        elif self.noise_range != af.nan and self.noise_range < 0:
            sys.exit("noise_range has to be a positive number")
        elif self.clamp_strength != af.nan and (self.clamp_strength > 1 or self.clamp_strength < 0):
            sys.exit("clamp_strength should be between 0 and 1")

    def increment_outputderiveCache(self, input_matrix):
        """
        Increments the output derivative cache.

        Args:
            input_matrix (ndarray): The matrix containing input derivatives.
        """
        self.outputderivCache = self.outputderivCache + input_matrix
