import bz2
import gzip
import copy
import math
import sys
import warnings
import random

from .group import Group
from .stats_plotter import StatsPlotter
from .link.link_factory import LinkFactory
from PyLens.examples.example_set import ExampleSet
from PyLens.examples.example_iterator import ExampleIterator
from .link.link_one_to_one import LinkOneToOne
from .error_functions.error import Error
from .error_functions.squared_error import SquaredError
from .error_functions.cross_entropy_error import CrossEntropyError
from .error_functions.cosine_error import CosineError
from .error_functions.divergence_error import DivergenceError
from .unit_cost_functions.conv_quad_cost import ConvQuadCost
from .unit_cost_functions.cosine_cost import CosineCost
from .unit_cost_functions.linear_cost import LinearCost
from .unit_cost_functions.logistic_cost import LogisticCost
from .unit_cost_functions.quadratic_cost import QuadraticCost
from .optimizer.steepest_optimizer import SteepestOptimizer
from .optimizer.momentum_optimizer import MomentumOptimizer
from .optimizer.dougsmomentum_optimizer import DougsMomentumOptimizer
from .optimizer.deltabardelta_optimizer import DeltaBarDeltaOptimizer
from .optimizer.adam_optimizer import AdamOptimizer
from .parameters import NetworkParameters
from .parameters import ExampleParameters
from .parameters import OptimizerParameters
from .array_factory import Array_factory as af
from .parallel_training import create_parallel_network
from time import sleep, time
import time as _time
if sys.platform != "darwin":
    import keyboard
else:
    keyboard = None
from itertools import product
import json
import codecs
import os
from threading import Thread, Event
from humanfriendly import format_timespan
import sympy as sp
import pickle
from pprint import pprint
import logging
import ray
import warnings

network_params = NetworkParameters()
example_params = ExampleParameters()
optimizer_params = OptimizerParameters()

SYM_ERROR_MESSAGES = {
    sp.zoo: "Error: Division by zero (Complex Infinity)",
    sp.nan: "Error: Undefined value (Not a Number)",
    sp.oo: "Error: Approaches positive infinity",
    -sp.oo: "Error: Approaches negative infinity"
}

class Network:
    """ 
    The Network class is used to build the architecture of standard neural networks
    and run examples through the network.  It also serves as the base class for other networks.
    
    """
    def __init__(self, name,
                 baseType='numpy',
                 time_intervals=network_params.PAR_N_numTimeIntervals,
                 ticks_per_interval=network_params.PAR_N_numTicksPerInterval,
                 add_bias=True,
                 learning_rate=None,
                 batch_error_threshold=network_params.PAR_N_criterion,
                 group_criterion_threshold=network_params.PAR_N_trainGroupCrit,
                 num_updates=network_params.PAR_N_numUpdates,
                 min_criterion_batches=network_params.PAR_N_minCritBatches,
                 update_method=network_params.PAR_N_algorithm,
                 stats_plotted=None,
                 pseudo_example_freq=None,
                 graph_title="error vs Report Interval",
                 graph_quantity="error",
                 tk_root=None, 
                 net_res_save_path=None, 
                 output_cost_strength=optimizer_params.PAR_O_outputCostStrength, 
                 output_cost_peak=optimizer_params.PAR_O_outputCostPeak,
                 debug=False):

        """
        Args:
            name (str): name of Network
            time_intervals (int): number of time steps needed for each example run on the network
            input_groups (List of Group objects): list of groups inputting into the network
            output_groups (List of Group objects): list of groups outputting from the network
            groups (list of Group objects): list of all groups in network - in order
            learning_rate (int): learning rate of the network
            example_sets (Example_Set): list of Example Set objects
            training_set (Example_Set): Example set used for training
            testing_set (Example_Set): Example set used for testing
            bias (Group): The standard bias group
        """

        self.unit_to_plot = None
        self.group_to_plot = None

        if not hasattr(self, 'network_type'):
            self.network_type = 'standard'

        self.name = name
        self.time_intervals = time_intervals
        self.input_groups = []
        self.output_groups = []
        self.cost_functions = []
        self.unit_cost_functions = []
        self.groups = []
        self.example_sets = []
        self.training_set = None
        self.testing_set = None
        self.bias = None
        self.max_example_time = 0
        self.errors = None
        self.batch_errors = None  # errors accumulated over each batch
        self.error_derivs = None
        self.unit_cost = None
        self.unit_cost_derivs = None
        self.batch_unit_costs = None # unit cost accumulated over each batch
        self.network_error = 0
        self.test_errors = None
        self.batch_test_errors = None
        self.test_error_derivs = None
        self.test_unit_cost = None
        self.batch_test_unit_cost = None
        self.test_unit_cost_derivs = None
        self.res = ""
        self.simulator = None
        # if it's in GUI mode, this will be set to true in the Simulator class
        self.visualized = False  # whether the training is in GUI mode as opposed to background mode
        self.last_example_trained = None  # the last example set trained
        self.plot = True  # whether train function should plot errors and weight cost
        # Used in standard_train_net function
        self.parallel_workers = None # workers on each CPU core
        self.initialization_actions = []

        # Flag to use multiprocessing
        # Initialized to False, to be changed in train method
        self.parallel_mode = False
        self.num_worker = 2 # number of workers, right now just being set to 2


        self.num_update = 0
        self.criterion_reached = False
        
        self.error_functions = {"cross_entropy": CrossEntropyError, "squared": SquaredError,
                                "cosine": CosineError, "divergence": DivergenceError}
        self.unit_cost_functions_map = {"conv_quad": ConvQuadCost, "cosine": CosineCost,
                                "linear": LinearCost, "logistic": LogisticCost, 
                                "quadratic": QuadraticCost}

        self.gain = network_params.PAR_N_gain
        self.ternary_shift = network_params.PAR_N_ternaryShift
        self.rand_range = network_params.PAR_N_randRange
        self.noise_range = network_params.PAR_N_noiseRange
        self.clamp_strength = network_params.PAR_N_clampStrength
        self.initInput = network_params.PAR_N_initInput
        self.initOutput = network_params.PAR_N_initOutput
        self.initOutputBias = network_params.PAR_N_initOutputBias
        self.baseType = baseType
        self.current_tick = 0
        self.ticks_per_interval = ticks_per_interval
        self.max_ticks = self.ticks_per_interval * self.time_intervals
        self.dt = 1 / ticks_per_interval
        self.batch_error_threshold = batch_error_threshold
        self.num_groups = 0
        self.num_units = 0
        self.num_inputs = 0
        self.num_outputs = 0
        self.criterion_reached = False
        self.user_interrupt = False
        self.group_criterion_reached = False
        self.group_criterion_threshold = group_criterion_threshold
        self.test_group_criterion_reached = False
        self.test_group_criterion_threshold = network_params.PAR_N_testGroupCrit
        self.test_error_criterion = False
        self.batch_test_errors_threshold = 0.0
        self.num_updates = num_updates
        self.min_criterion_batches = min_criterion_batches
        self.rate_increment = 0
        self.rate_decrement = 1
        self.paused = False
        self.pseudoExampleFreq = pseudo_example_freq
        self.stop_event = Event()
        self.net_res_save_path = net_res_save_path
        self.add_bias = add_bias
        self.output_cost_strength = output_cost_strength
        self.output_cost_peak = output_cost_peak
        self.target_radius = optimizer_params.PAR_O_targetRadius
        self.zero_error_radius = optimizer_params.PAR_O_zeroErrorRadius

        self.stats_plotter = StatsPlotter(self, points_on_plot=0,
                                          stats_plotted=None, plot_colors=None,
                                          report_interval=1)
        self.set_update_method(learning_rate, update_method, printout=False)
        self.learning_rate = self.optimizer.learning_rate


        if self.add_bias:
            self.add_group(1, name="bias", group_type="bias", input_transforms=[], output_transforms=["bias_clamp"])

        self.training_sets = []
        self.testing_sets = []
        self.loaded_example_sets = {}
        self.points_on_plot = 0
        if stats_plotted is None:
            self.stats_plotted = ["error"]
        else:
            self.stats_plotted = stats_plotted
        self.report_interval = 1
        self.plot_colors = ["blue", "orange", "green", "red", "purple", "brown", "pink" "gray", "olive", "cyan"]
        self.live_update = True
        self.print_reports = True
        self.run_number = 0
        self.runs = []
        self.use_keyboard = False
        self.ticks_on_example = 0

        self.clock_time = 0
        self.plot_update_time = 0.5
        self.plot_thread_created = False

        self.graphs = [] # Empty list of graph_viewers from gui
        self.examples_token_trained = 0


        # Some extra attributes for debugging
        self.debug = debug
        self.debug_errors = []
        self.debug_weights = []
        self.link_types = []
        self.links_dict = {}


    def _resolve_groups(self, group_names):
        """
        Helper: turn '*', None, a single name, or a list of names into a list of Group objects.
        """
        if group_names is None or group_names == "*":
            return list(self.groups)
        if isinstance(group_names, str):
            return [self.get_group_by_name(group_names)]
        return [self.get_group_by_name(n) for n in group_names]
    
    def _open_text(self, filename: str, mode: str):
        """Open a text file, possibly compressed.

        Args:
            filename (str): The name of the file to open.
            mode (str): The mode in which to open the file.

        Returns:
            _type_: _description_
        """
        if filename.endswith(".gz"):
            return gzip.open(filename, mode + "t")
        if filename.endswith(".bz2"):
            return bz2.open(filename, mode + "t")
        return open(filename, mode, encoding="utf-8")


    def set_properties(self, **updates):
        """
        set properties for the network and its objects. This is required to set the parallel worker.
        updates are sequences of (target_object, new value) pairs
        """
        def _set_nested(target_object, updates):
            """Helper that can recursively set attributes for nested dicts/lists."""
            for property_name, new_value in updates.items():

                if isinstance(new_value, dict):
                    # if the object already has this attribute, dive deeper
                    if hasattr(target_object, property_name):
                        _set_nested(getattr(target_object, property_name), new_value)
                    else:
                        setattr(target_object, property_name, new_value)

                elif isinstance(new_value, list) and hasattr(target_object, property_name):
                    lst = getattr(target_object, property_name)
                    for i, subupdates in enumerate(new_value):
                        if isinstance(subupdates, dict) and i < len(lst):
                            _set_nested(lst[i], subupdates)

                else:
                    setattr(target_object, property_name, new_value)

        self.record_action("set_properties", **updates)
        _set_nested(self, updates)
        return self

    def record_action(self, method_name, *args, **kwargs):
        """
        Store the method name and its parameters in a serializable way
        """
        # Store the method name and its parameters in a serializable way
        action = json.dumps({
            "method": method_name,
            "args": args,
            "kwargs": kwargs
        })
        self.initialization_actions.append(action)

    def replay_initialization_for_workers(self, initialization_actions):
        """
        Reinitializes workers or agents for replay purposes, preparing them for replay tasks or simulations
        """
        for action_str in initialization_actions:
            action = json.loads(action_str)
            # Retrieve the method and call it with stored arguments and keyword arguments
            getattr(self, action['method'])(*action['args'], **action['kwargs'])

    def set_optimizer_weight_decay(self, weight_decay):
        self.optimizer.weight_decay = weight_decay
    def set_optimizer_params_PAR_O_zeroErrorRadius(self, PAR_O_zeroErrorRadius):
        self.optimizer.optimizer_params.PAR_O_zeroErrorRadius = PAR_O_zeroErrorRadius

    def get_stats_plotter(self):
        print("Returning stats_plotter")
        return self.stats_plotter

    def set_update_method(self, lr, update_method="steepest", printout=True):
        """
        Set the weight update method to update_method
        
        Args:
            lr (float): The learning rate to be used during optimization.
            update_method (str): Choices are: "steepest", "momentum", "dougs momentum", "delta bar delta", "adam"
        """
        update_method_list = ["steepest", "momentum", "dougs momentum", "delta bar delta", "adam"]

        self.update_method = update_method.lower()

        # if weight update algorithm is not specified, use default settings
        if self.update_method not in update_method_list:
            print(f"Update method of {update_method} not recognized")
            self.update_method = network_params.PAR_N_algorithm.lower()

        if self.update_method == "steepest":
            self.optimizer = SteepestOptimizer(self, lr)
        elif self.update_method == "momentum":
            self.optimizer = MomentumOptimizer(self, lr)
        elif self.update_method == "dougs momentum":
            self.optimizer = DougsMomentumOptimizer(self, lr)
        elif self.update_method == "delta bar delta":
            self.optimizer = DeltaBarDeltaOptimizer(self, lr)
        elif self.update_method == "adam":
            self.optimizer = AdamOptimizer(self, lr)
        else:
            self.optimizer = SteepestOptimizer(self, lr)
        self.learning_rate = lr
        if printout:
            print(f"Initialized {self.update_method} optimizer with learning rate of {self.learning_rate}")

    def update_learning_rate(self, learning_rate):
        """
        Updates the learning rate of the current optimizer and the network
        
        Args:
            learning_rate (float)
        """
        self.optimizer.learning_rate = learning_rate
        self.learning_rate = self.optimizer.learning_rate

    def update_optimizer_momentum(self, momentum):
        """
        Updates the momentum of the current optimizer
        
        Args:
            momentum (float)
        """
        self.optimizer.momentum = momentum
    
    def update_optimizer_weight_decay(self, weight_decay):
        """
        Updates the weight decay of the current optimizer
        
        Args:
            weight_decay (float)
        """
        self.optimizer.weight_decay = weight_decay
    
    def update_target_radius(self, target_radius):
        """
        Updates the target radius of the every cost function
        
        Args:
            target_radius (float)
        """
        self.target_radius = target_radius
        for cost_function in self.cost_functions:
            cost_function.target_radius = target_radius

    def update_zero_error_radius(self, zero_error_radius):
        """
        Updates the zero error radius of the every cost function
        
        Args:
            zero_error_radius (float)
        """
        self.zero_error_radius = zero_error_radius
        for cost_function in self.cost_functions:
            cost_function.zero_error_radius = zero_error_radius

    def toggle_live_update(self, live_update=True):
        """
        Turn live update on or off
        
        Args:
            live_update (boolean)
        """
        self.live_update = live_update
        self.stats_plotter.toggle_live_update(live_update)

    def toggle_print_reports(self, print_reports=True):
        """
        Turn print reports on or off
        
        Args:
            print_reports (boolean)
        """
        self.stats_plotter.toggle_print_reports(print_reports)

    def toggle_plots(self, plots=True):
        """
        Turn plotting on or off
        
        Args:
            plots (boolean)
        """
        self.plot = plots

    def toggle_keyboard(self, use=True):
        """
        Disable or enable keyboard input
        Used for github action test
        
        Args:
            use (boolean)
        """
        self.use_keyboard = use

    def add_plotting_stats(self, stats_plotted=None):
        """
        Add a stat to be included in the live plots
        
        Args:
            stats_plotted (boolean)
        """
        self.stats_plotter.add_plotting_stats(stats_plotted)

    def set_plotting_stats(self, stats_plotted=None):
        """
        Set which stats are to be included in the live plots
        
        Args:
            stats_plotted (boolean)
        """
        self.stats_plotted = stats_plotted

    def set_plot_colors(self, plot_colors):
        """
        Sets the colors to be used in the plots
        
        Args:
            plot_colors (List[str]): list of colors used in plots, in the order of the runs
        """
        self.plot_colors = plot_colors

    def add_group(self,
                  num_units,
                  name=None,
                  group_type="hidden",
                  input_transforms=None,
                  output_transforms=None,
                  error_function=None,
                  lesion_rate=None,
                  dropout_rate=None,
                  num_cols=None,
                  biased=None,
                  unit_cost_function=None):
        """
        Add a group to the existing network

        Args:
            num_units (int): number of units in the group
            name (str): name of the group
            group_type (str): defines the group type (input, output or bias)
            input_transforms (List of Input_Transform objects): list of transforms to apply to the inputted data
            output_transforms (List of Output_Transform objects): list of transforms to apply to the outputted data
            error_function: the function used to calculate the error for the group during training.
            lesion_rate (float): rate at which units in the group are lesioned or disabled.
            dropout_rate (float): probability of dropping out a unit in the group
            num_cols (int): number of columns in the group
            biased (boolean): whether the group is biased or not
            unit_cost_function: the function to cocmpute the cost for each unit in the group
        """
        self.record_action("add_group",
                           num_units=num_units,
                           name=name,
                           group_type=group_type,
                           input_transforms=input_transforms,
                           output_transforms=output_transforms,
                           error_function=error_function,
                           lesion_rate=lesion_rate,
                           dropout_rate=dropout_rate,
                           num_cols=num_cols,
                           biased=biased,
                           unit_cost_function=unit_cost_function)

        new_group = None
        if name is None:
            name = "{}_{}".format(group_type, len(self.groups))
        if self.network_type == 'boltzmann':
            if input_transforms is None or len(input_transforms) == 0:
                if group_type == "input":
                    input_transforms = []
                elif group_type != "bias":
                    input_transforms = ["boltzmann"]
            else:
                if group_type not in ("input", "bias"):
                    if input_transforms[0] != "boltzmann":
                        input_transforms.insert(0, "boltzmann")

            if output_transforms is None or len(output_transforms) == 0:
                if group_type == "input":
                    output_transforms = []
                elif group_type != "bias":
                    output_transforms = ["boltzmann"]
            else:
                if group_type not in ("input", "bias"):
                    if output_transforms[0] != "boltzmann":
                        output_transforms.insert(0, "boltzmann")
        else:
            if input_transforms is None:
                if group_type == "input":
                    input_transforms = []
                elif group_type == "hidden":
                    input_transforms = ["dot"]
                elif group_type == "output":
                    input_transforms = ["dot"]
            if output_transforms is None:
                if group_type == "input":
                    output_transforms = []
                elif group_type == "hidden":
                    output_transforms = ["sigmoid"]
                elif group_type == "output":
                    # raise Error("You need to specify the output transformation of output layer")
                    output_transforms = ["sigmoid"]
        # check if name exists
        if self.check_name(name):
            # instantiate a new group object, append it to master list of groups
            new_group = Group(name, num_units, group_type, input_transforms, output_transforms, self.time_intervals,
                                  self.ticks_per_interval, self, dropout_rate=dropout_rate,
                                  num_cols=num_cols)

            if (group_type != "input" and group_type != "bias" and group_type != "elman" and biased is None):
                biased = True
            elif ((group_type == "input" or group_type == "elman") and biased is None):
                biased = False
            if (group_type != "bias" and self.bias is not None and biased):
                new_group.add_bias(self.bias)
            self.num_groups += 1
            self.num_units += num_units

            self.groups.append(new_group)

            # check if the group is of input or output type, and append it to appropriate array
            if group_type == "input":
                self.input_groups.append(new_group)
            elif group_type == "output":
                # append error functions
                if error_function is None not in list(self.error_functions.keys()):
                    if self.network_type == "boltzmann":
                        error_function = "cross_entropy"
                        error_function = self.error_functions[error_function](new_group)
                    else:
                        if output_transforms[-1] == "sigmoid" or output_transforms[-1] == "soft_max":
                            error_function = "cross_entropy"
                            error_function = self.error_functions[error_function](new_group)
                        elif output_transforms[-1] == "soft_max":
                            error_function = "divergence_error"
                            error_function = self.error_functions[error_function](new_group)
                        elif output_transforms[-1] == "linear" or output_transforms[-1] == "hard_clamp":
                            error_function = "squared"
                            error_function = self.error_functions[error_function](new_group)
                else:
                    error_function = self.error_functions[error_function](new_group)
                self.cost_functions.append(error_function)
                if unit_cost_function is None not in list(self.unit_cost_functions_map.keys()):
                    unit_cost_function = None
                else:
                    unit_cost_function = self.unit_cost_functions_map[unit_cost_function](new_group)
                self.unit_cost_functions.append(unit_cost_function)
                self.output_groups.append(new_group)
            elif group_type == "bias":
                self.bias = new_group
                self.bias.initOutput = af.array([self.initOutputBias])
                self.bias.output_matrix = self.bias.initOutput
    

    def get_group_by_name(self, group_name):
        """
        Retrieves a group from the network by its name.
        
        Args:
            group_name (str): the name of the group to search for
            
        Returns:
            Group: the group object with the specified name
        """
        for group in self.groups:
            if group.name == group_name:
                return group
        raise Error("No group in current architecture match the given group name")

    def _iter_incoming_links(self, groups=None, link_type: str | None = None):
        """
        Yield incoming links for selected destination groups, optionally filtered by link_type.
        """
        target_groups = self._resolve_groups(groups) if groups is not None else list(self.groups)
        for g in target_groups:
            for link in getattr(g, "incoming_links", []) or []:
                if link_type is not None and getattr(link, "link_type", None) != link_type:
                    continue
                yield link

    def _apply_freeze_to_links(
        self,
        links,
        *,
        do_freeze: bool,
        all_links: bool = False,
        unit_indices=None,
        link_indices=None,
    ) -> int:
        """
        Apply freeze/thaw on a set of links.
        Returns number of links visited.
        """
        has_selector = (unit_indices is not None) or (link_indices is not None)
        visited = 0
        for link in links:
            visited += 1
            if all_links or not has_selector:
                if do_freeze:
                    link.freeze_incoming_links(freeze_all=True)
                else:
                    link.thaw_incoming_links(thaw_all=True)
                continue
            if do_freeze:
                if unit_indices is not None:
                    link.freeze_incoming_links(unit_indices=unit_indices)
                if link_indices is not None:
                    link.freeze_incoming_links(link_indices=link_indices)
            else:
                if unit_indices is not None:
                    link.thaw_incoming_links(unit_indices=unit_indices)
                if link_indices is not None:
                    link.thaw_incoming_links(link_indices=link_indices)
        return visited

    def freeze_links(
        self,
        groups=None,
        units_indices=None,
        link_indices=None,
        link_type: str | None = None,
        freeze_all: bool = False,
    ) -> int:
        """
        Freeze weights on links selected by destination group(s), indices, and optional link_type.
        If no selectors are provided, freezes all selected links.
        """
        links = list(self._iter_incoming_links(groups=groups, link_type=link_type))
        return self._apply_freeze_to_links(
            links,
            do_freeze=True,
            all_links=freeze_all,
            unit_indices=units_indices,
            link_indices=link_indices,
        )

    def thaw_links(
        self,
        groups=None,
        units_indices=None,
        link_indices=None,
        link_type: str | None = None,
        thaw_all: bool = False,
    ) -> int:
        """
        Thaw weights on links selected by destination group(s), indices, and optional link_type.
        If no selectors are provided, thaws all selected links.
        """
        links = list(self._iter_incoming_links(groups=groups, link_type=link_type))
        return self._apply_freeze_to_links(
            links,
            do_freeze=False,
            all_links=thaw_all,
            unit_indices=units_indices,
            link_indices=link_indices,
        )

    # cLens-aligned aliases
    def freeze_weights(self, groups=None, units_indices=None, link_indices=None, link_type: str | None = None):
        return self.freeze_links(
            groups=groups,
            units_indices=units_indices,
            link_indices=link_indices,
            link_type=link_type,
        )

    def thaw_weights(self, groups=None, units_indices=None, link_indices=None, link_type: str | None = None):
        return self.thaw_links(
            groups=groups,
            units_indices=units_indices,
            link_indices=link_indices,
            link_type=link_type,
        )

    def freeze_group_inputs(self, incoming_group=None, freeze_all=None, units_indices=None, link_indices=None, bias_indices=None, link_type=None):
        """
        Freezes the input links (connections) to a specified group in the network.

        Freeze the specified input links in units_indices to this group.
        If units_indices is a 1D np.array then freeze all the links for each for the unit_indices
        for this group.
        If link_indices is an list with (i,j) pairs then freeze all the links connecting the ith
        unit from previous group to the jth unit in this group.
        If bias is not None then freeze the incoming bias links as well. Must specify bias_indices
        
        Args:
            incoming_group (str): the name of the group that the links are coming from
            freeze_all (boolean): whether to freeze all the links in the network
            units_indices (List[int]): the indices of the units in the incoming group to freeze
            link_indices (List[Tuple[int, int]]): each pair specifies the link in this group to be frozen
            bias_indices (List[int]): the indices of the bias units to freeze
            
        Raises:
            Error: if the incoming group does not exist in the network
        """
        if freeze_all:
            self.freeze_all_links(link_type=link_type)
            return
        if incoming_group is None:
            raise Error("incoming_group is required unless freeze_all=True")
        g1 = self.get_group_by_name(incoming_group)
        # Preserve legacy behavior:
        # - main link path uses the last incoming link
        # - bias path uses the first incoming link
        target_main = []
        if getattr(g1, "incoming_links", None):
            main_link = g1.incoming_links[-1]
            if link_type is None or getattr(main_link, "link_type", None) == link_type:
                target_main.append(main_link)
        self._apply_freeze_to_links(
            target_main,
            do_freeze=True,
            all_links=False,
            unit_indices=units_indices,
            link_indices=link_indices,
        )
        if bias_indices and getattr(g1, "incoming_links", None):
            bias_link = g1.incoming_links[0]
            if link_type is None or getattr(bias_link, "link_type", None) == link_type:
                self._apply_freeze_to_links(
                    [bias_link],
                    do_freeze=True,
                    all_links=False,
                    unit_indices=bias_indices,
                )

    def freeze_all_links(self, link_type: str | None = None):
        """
        Freeze all the links in the network.
        """
        return self.freeze_links(link_type=link_type, freeze_all=True)

    def thaw_all_links(self, link_type: str | None = None):
        """
        Thaw all the links in the network.
        """
        return self.thaw_links(link_type=link_type, thaw_all=True)

    def thaw_group_inputs(self, incoming_group=None, thaw_all=None, units_indices=None, link_indices=None, bias_indices=None, link_type=None):
        """
        Thaw or unfreeze the specified links in the network.
        
        Thaw_all will unfreeze all the links in the network
        Thaw the specified input links in units_indices to this group.
        If units_indices is a 1D np.array then thaw all the links for each for the unit_indices
        for this group.
        If link_indices is an list with (i,j) pairs then thaw all the links connecting the ith
        unit from previous group to the jth unit in this group.
        If bias is not None then thaw the incoming bias links as well. Must specify bias_indices
        
        Args:
            incoming_group (str): the name of the group that the links are coming from
            thaw_all (boolean): whether to thaw all the links in the network
            units_indices (List[int]): the indices of the units in the incoming group to thaw
            link_indices (List[Tuple[int, int]]): each pair specifies the link in this group to be thawed
            bias_indices (List[int]): the indices of the bias units to thaw
            
        Raises:
            Error: if the incoming group does not exist in the network
        """
        if thaw_all:
            self.thaw_all_links(link_type=link_type)
            return
        if incoming_group is None:
            raise Error("incoming_group is required unless thaw_all=True")
        g1 = self.get_group_by_name(incoming_group)
        target_main = []
        if getattr(g1, "incoming_links", None):
            main_link = g1.incoming_links[-1]
            if link_type is None or getattr(main_link, "link_type", None) == link_type:
                target_main.append(main_link)
        self._apply_freeze_to_links(
            target_main,
            do_freeze=False,
            all_links=False,
            unit_indices=units_indices,
            link_indices=link_indices,
        )
        if bias_indices and getattr(g1, "incoming_links", None):
            bias_link = g1.incoming_links[0]
            if link_type is None or getattr(bias_link, "link_type", None) == link_type:
                self._apply_freeze_to_links(
                    [bias_link],
                    do_freeze=False,
                    all_links=False,
                    unit_indices=bias_indices,
                )

    def lesion_all_groups(self, lesion_rate):
        """
        Applies a lesion to all groups in the network.
        
        See Also:
            def add_group: For the description of 'lesion_rate'.
        """
        for group in self.groups:
            group.lesion_group(lesion_rate)

    def lesion_specific_group(self, group_name, lesion_rate=0, lesion_units=None):
        """
        Applies a lesion to a specific group in the network.
        
        Args:
            group_name (str): the name of the group to lesion
            lesion_rate (float): the rate at which the units in the group are lesioned
            lesion_units (List[int]): the indices of the units in the group to lesion
        """
        assert (group_name in [group.name for group in self.groups])

        group = self.get_group_by_name(group_name)
        if lesion_rate:
            group.lesion_group(lesion_rate)

        if lesion_units:
            group.lesion_units(lesion_units)

    def heal_all_groups(self, p=None):
        """
        Heals all the groups in the network by either restoring all lesioned units or 
        restoring a proportion of units based on the given proportion `p`.
        
        Args:
            p (float): the proportion of units to heal in the group
        """
        if not p:
            for group in self.groups:
                group.heal()
        else:
            for group in self.groups:
                group.heal_by_proportion(p)

    def heal_specific_group(self, group_name, heal_all=False, heal_units=None,
                            heal_rate=None):
        """
        Heals a specific group by either healing all units, healing specific units, 
        or healing a proportion of lesioned units based on `heal_rate`.
        
        Args:
            group_name (str): the name of the group to heal
            heal_all (boolean): whether to heal all units in the group
            heal_units (List[int]): the indices of the units in the group to heal
            heal_rate (float): the proportion of units to heal in the group
        """
        assert (group_name in [group.name for group in self.groups])

        group = self.get_group_by_name(group_name)
        if heal_all:
            group.heal()
        elif heal_units:
            group.heal_units(heal_units)
        elif heal_rate:
            group.heal_by_proportion(heal_rate)

    def lesion_all_links(self, lesion_rate):
        """
        Applies a lesion to all incoming links in all groups of the network.
        
        See Also:
            def add_group: For the description of 'lesion_rate'.
        """
        for group in self.groups:
            for link in group.incoming_links:
                link.lesion_link(lesion_rate)

    def link_specific_lesion(self, outgoing_group, incoming_group, proj_type,
                             cartesian=False, lesion_rate=0, links_to_lesion=None):
        """
        Applies a lesion to specific links between two groups in the network.
        
        Args:
            outgoing_group (str): the name of the group passing the data forward
            incoming_group (str): the name of the group receiving data
            proj_type (str): the type of projection between the two groups
            cartesian (boolean): whether the indices are cartesian or not
            lesion_rate (float): the rate at which the links are lesioned
            links_to_lesion (List[int]): the indices of the links to lesion
        """
        g1, g2 = self._group_pair_from_names(outgoing_group, incoming_group)
        link = g2.incoming_links[-1]
        if lesion_rate:
            link.lesion_link(lesion_rate)
        if links_to_lesion:
            ltl = self.indices_to_lesion(links_to_lesion, proj_type, cartesian)
            link.specific_lesion_link(ltl)

    def lesion_bias_links(self, group_name=None, lesion_rate=0):
        """
        Applies a lesion to the bias links in the network.
        
        Args:
            group_name (str): the name of the group to lesion
    
        See Also:
            def add_group: For the description of 'lesion_rate'.
        """
        if group_name:
            group = self.get_group_by_name(group_name)
            if self.bias and len(group.incoming_links) > 1:
                bias_link = group.incoming_links[0]
                bias_link.lesion_link(lesion_rate)
        else:
            for group in self.groups:
                if self.bias and len(group.incoming_links) > 1:
                    bias_link = group.incoming_links[0]
                    bias_link.lesion_link(lesion_rate)

    def heal_bias_links(self, group_name=None, heal_rate=None):
        """
        Heals the bias links in the network.
        
        Args:
            group_name (str): the name of the group to heal
            heal_rate (float): the rate at which the bias links are healed
        """
        if group_name:
            group = self.get_group_by_name(group_name)
            if self.bias and len(group.incoming_links) > 1:
                bias_link = group.incoming_links[0]
                if heal_rate:
                    bias_link.heal_by_proportion(heal_rate)
                else:
                    bias_link.heal()
        else:
            for group in self.groups:
                if self.bias and len(group.incoming_links) > 1:
                    bias_link = group.incoming_links[0]
                    if heal_rate:
                        bias_link.heal_by_proportion(heal_rate)
                    else:
                        bias_link.heal()

    def add_all_links_noise(self, noise_type="uniform", p=None, z=1):
        """
        Add noise to all that are at most z standard deviation away from the mean of the weight entries
        and has noise_type distribution.
        
        Args:
            noise_type (str): The type of noise distribution to apply (e.g., "uniform", "normal"). Default is "uniform".
            p (float): the proportion of links to add noise to
            z (float): the standard deviation of the noise
        """
        for group in self.groups:
            for link in group.incoming_links:
                link.add_noise(noise_type, p, z)

    def add_links_specific_noise(self, outgoing_group, incoming_group, noise_type="uniform", p=None, z=1):
        """
        Add noise to specific links between two groups in the network.
        
        Args:
            outgoing_group (str): the name of the group passing the data forward
            incoming_group (str): the name of the group receiving data
            noise_type (str): the type of noise distribution to apply (e.g., "uniform", "normal"). Default is "uniform".
            p (float): the proportion of links to add noise to
            z (float): the standard deviation of the noise
        """
        g1, g2 = self._group_pair_from_names(outgoing_group, incoming_group)
        for link in g2.incoming_links:
            link.add_noise(noise_type, p, z)

    def remove_all_links_noise(self):
        """
        Remove noise from all links in the network.
        """
        for group in self.groups:
            for link in group.incoming_links:
                link.remove_noise()

    def remove_links_specific_noise(self, outgoing_group, incoming_group):
        """
        Remove noise from specific links between two groups in the network.
        
        See Also:
            def links_specific_lesion: For the description of 'outgoing_group' and 'incoming_group'.
        """
        g1, g2 = self._group_pair_from_names(outgoing_group, incoming_group)
        for link in g2.incoming_links:
            link.remove_noise()


    def heal_all_links(self, heal_rate=None):
        """
        Heals all the links in the network by either restoring all lesioned links or
        restoring a proportion of links based on the given proportion `heal_rate`.
        
        Args:
            heal_rate (float): the proportion of links to heal
        """
        for group in self.groups:
            for link in group.incoming_links:
                if heal_rate is None:
                    link.heal()
                else:
                    link.heal_by_proportion(heal_rate)

    def link_specific_heal(self, outgoing_group, incoming_group, proj_type,
                           cartesian=False, heal_mask=False, links_to_heal=None, heal_rate=None):
        """
        Heals specific links between two groups in the network.
        
        Args:
            heal_mask (boolean): whether to heal all links in the mask
            links_to_heal (List[int]): the indices of the links to heal
            heal_rate (float): the proportion of links to heal
            
        See Also:
            def links_specific_lesion: For the description of 'outgoing_group', 'incoming_group',
            'proj_type', and 'cartesian'.
        """
        g1, g2 = self._group_pair_from_names(outgoing_group, incoming_group)
        link = g1.outgoing_links[0]
        if heal_mask:
            link.heal()
        elif links_to_heal:
            ltl = self.indices_to_lesion(links_to_heal, proj_type, cartesian)
            link.link_specific_heal(ltl)
        elif heal_rate:
            link.heal_by_proportion(links_to_heal)

    @staticmethod
    def indices_to_lesion(indices, proj_type, cartesian=False):
        """
        Heal lesion connections
        
        - If cartesian is False, then indices should be passed as a list of (i,j) pairs to heal.
        - If cartesian is True, then indices should be passed as a list of 2 lists, one dedicated for index i and one for index j
            - Each list should contain tuples.
            - The indices to lesion is the cartesian product of the elements in the kth tuple in the first list and second list
            - Example: indices = [[(1, 2, 3), (4,)], [(5,), (6, 7)]]
            -          so the (i, j) pairs are: (1, 5), (2, 5), (3, 5), (4, 6), (4, 7)
        
        Args:
            indices (List): the indices of the links to lesion
            
        See Also:
            def links_specific_lesion: For the description of 'cartesian' and 'proj_type'.
        """
        if indices and not cartesian:
            assert (isinstance(indices, list))
            if proj_type == "one-to-one":
                assert (isinstance(i, int) for i in indices)
            else:
                assert (len(pair) == 2 for pair in indices)
            assert (isinstance(pair, tuple) and isinstance(pair[0], int) and isinstance(pair[1], int) for pair in
                    indices)
            return indices

        elif indices and cartesian:

            assert (isinstance(indices, list) and len(indices) == 2)
            assert (isinstance(lst, list) for lst in indices)
            assert ((isinstance(lst, tuple) for lst in indices[0]) and (isinstance(lst, tuple) for lst in indices[1]))
            assert (len(indices[0]) == len(indices[1]))
            final_indices = []

            for k in range(len(indices[0])):
                final_indices.extend(list(product(list(indices[0][k]), list(indices[1][k]))))

            return final_indices
        return None

    def check_name(self, name):
        """
        Checks if the given name is available in the network groups.
        
        Args:
            name (str): the name to check
        """
        result = True

        for group in self.groups:
            if group.name == name:
                result = False

        return result

    def _group_pair_from_names(self, outgoing_group, incoming_group):
        """
        Connect two groups together in the network

        See Also:
            def links_specific_lesion: For the description of 'outgoing_group' and 'incoming_group'.
            
        Returns:
            g1, g2: the group reference
        """

        g1, g2 = None, None

        for group in self.groups:
            if group.name == outgoing_group:
                g1 = group
            if group.name == incoming_group:
                g2 = group
        return g1, g2
    
    def elman_connect(self, source_group, context_group):
        """
        Connect a source group to an ELMAN context group by wiring the Elman_Clamp's source.
        No links are created; the clamp adds source outputs onto context outputs.

        Args:
            source_group (str|Group)
            context_group (str|Group)

        Raises:
            ValueError on invalid usage.
        """
        src, ctx = self._group_pair_from_names(source_group, context_group)

        if src.num_units < ctx.num_units:
            raise ValueError(
                f'Cannot elmanConnect "{src.name}" -> "{ctx.name}": '
                f'{src.num_units} < {ctx.num_units}.'
            )

        # Find an Elman_Clamp transform on the context group that is not already set
        clamp = None
        for t in getattr(ctx, "output_transforms", []):
            # rely on class name or attribute; adjust if you wrap transforms differently
            if t.name == "Elman_Clamp":
                clamp = t
                break
        if clamp is None:
            raise ValueError(f'Group "{ctx.name}" has no ELMAN_CLAMP output transform.')

        if clamp.source_group is not None:
            raise ValueError(f'Group "{ctx.name}" already has an Elman source set.')

        clamp.source_group = src

        return True

    def copy_connect(self, source_group, copy_group, field: str):
        """
        Wire a COPY transform on `copy_group` so it reads `field`
        from `source_group`.
        """
        src, dst = self._group_pair_from_names(source_group, copy_group)

        if src.num_units != dst.num_units:
            raise ValueError("source and copy groups must have the same number of units")

        field = field.strip()
        valid = {"inputs","externalInputs","outputs","targets","inputDerivs","outputDerivs"}
        if field not in valid:
            raise ValueError(f"bad field type: {field}")

        # find the first empty COPY slot
        transform = None
        # input-phase
        for t in getattr(dst, "input_transforms", []):
            if t.name == "In_Copy" and getattr(t, "source_group", None) is None:
                transform = t
                break
        # output-phase
        if transform is None:
            for t in getattr(dst, "output_transforms", []):
                if t.name == "Out_Copy" and getattr(t, "source_group", None) is None:
                    transform = t
                    break

        if transform is None:
            raise ValueError(f'Group "{dst.name}" has no empty *_COPY slots.')

        transform.source_group = src
        transform.source_field = field
    
    def connect_groups(self, 
                       outgoing_group, 
                       incoming_group, 
                       initialization="uniform", 
                       rand_mean=0, 
                       rand_range=None, 
                       proj_type="full", 
                       link_type=None, 
                       lesion_rate=None, 
                       dropout_rate=None, 
                       perma_lesion_rate=None, 
                       bidirectional=False):
        """
        Connect two groups together in the network

        Args:
            initialization (str): weight initialization method ('uniform', 'gaussian', or 'kaiming')
            rand_mean (float): the mean of the random weights
            rand_range (float): the range of the random weights
            proj_type (str): the projection type between the groups
            link_type (str): the type of link defined by the user
            dropout_rate (float): the probability of dropping out a unit in the group
            perma_lesion_rate (float): the rate at which the units in the group are permanently lesioned
            bidirectional (boolean): whether the connection is bidirectional or not
            
        See Also:
            def links_specific_lesion: For the description of 'outgoing_group', 'incoming_group',
            'proj_type', and 'lesion_rate'.
        """
        self.record_action("connect_groups", 
                           outgoing_group=outgoing_group, 
                           incoming_group=incoming_group, 
                           initialization=initialization, 
                           rand_mean=rand_mean, 
                           rand_range=rand_range, 
                           proj_type=proj_type, 
                           link_type=link_type,
                           lesion_rate=lesion_rate, 
                           dropout_rate=dropout_rate, 
                           perma_lesion_rate=perma_lesion_rate, 
                           bidirectional=bidirectional)
        if rand_range == None:
            # Use networks rand_range if user does not specify one
            rand_range = self.rand_range

        g1, g2 = self._group_pair_from_names(outgoing_group, incoming_group)

        if proj_type == "elman":
            return self.elman_connect(outgoing_group, incoming_group)

        new_link = LinkFactory.construct_link(g1, g2, initialization, 
                                              rand_mean=rand_mean, 
                                              rand_range=rand_range,
                                              proj_type=proj_type, 
                                              link_type=link_type,
                                              dropout_rate=dropout_rate,
                                              perma_lesion_rate=perma_lesion_rate)
        g1.link_next(new_link)
        g2.link_previous(new_link)

        if link_type is not None:
            if link_type not in self.links_dict.keys():
                self.links_dict[link_type] = [new_link]
            else:
                self.links_dict[link_type].append(new_link)
            if link_type not in self.link_types:
                self.link_types.append(link_type)

        if bidirectional:
            new_link = LinkFactory.construct_link(g2, g1, initialization, 
                                                  rand_mean=rand_mean, 
                                                  rand_range=rand_range, 
                                                  proj_type=proj_type, 
                                                  link_type=link_type,
                                                  dropout_rate=dropout_rate, 
                                                  perma_lesion_rate=perma_lesion_rate)
            g2.link_next(new_link)
            g1.link_previous(new_link)
    
    def connect_units(self,
                  outgoing_group: str,
                  outgoing_unit_idx: list,     # indices within outgoing group
                  incoming_group: str,
                  incoming_unit_idx: list,     # indices within incoming group
                  initialization: str = "uniform",
                  rand_mean: float = 0.0,
                  rand_range: float | None = None,
                  proj_type: str = "full",
                  link_type=None,
                  lesion_rate: float | None = None,
                  dropout_rate: float | None = None,
                  perma_lesion_rate: float | None = None,
                  bidirectional: bool = False):
        """
        Create (or reuse) a link between specific units in `outgoing_group` and `incoming_group`,
        then mask the link so only those pairs are active.

        If a link with the same `link_type` already connects these two groups, reuse it:
        - print a warning
        - overwrite its connection_mask with the new pairs

        Otherwise, construct a new link and register it as usual.
        """

        self.record_action("connect_units",
            outgoing_group=outgoing_group,
            outgoing_unit_idx=outgoing_unit_idx,
            incoming_group=incoming_group,
            incoming_unit_idx=incoming_unit_idx,
            initialization=initialization,
            rand_mean=rand_mean,
            rand_range=rand_range,
            proj_type=proj_type,
            link_type=link_type,
            lesion_rate=lesion_rate,
            dropout_rate=dropout_rate,
            perma_lesion_rate=perma_lesion_rate,
            bidirectional=bidirectional,
        )

        if rand_range is None:
            rand_range = self.rand_range

        # Resolve groups
        g_out, g_in = self._group_pair_from_names(outgoing_group, incoming_group)

        # ----- Check for existing link with same type between these groups -----
        existing_link = None
        if getattr(g_in, "incoming_links", None):
            for L in g_in.incoming_links:
                if getattr(L, "outgoing_group", None) is g_out and getattr(L, "link_type", None) == link_type:
                    existing_link = L
                    break

        # Use existing link (warn) or create a new one
        if existing_link is not None:
            print(f"Warning: link with type '{link_type}' already exists between "
                f"'{g_out.name}' -> '{g_in.name}'. Overwriting previous connection mask.")
            link = existing_link
            # Do NOT re-register adjacency or links_dict; we’re reusing the same object.
        else:
            # Construct the link via factory
            link = LinkFactory.construct_link(
                g_out, g_in, initialization,
                rand_mean=rand_mean,
                rand_range=rand_range,
                proj_type=proj_type,
                link_type=link_type,
                dropout_rate=dropout_rate,
                perma_lesion_rate=perma_lesion_rate
            )
            # Register adjacency & bookkeeping only for new links
            g_out.link_next(link)
            g_in.link_previous(link)
            if link_type not in self.links_dict:
                self.links_dict[link_type] = [link]
            else:
                self.links_dict[link_type].append(link)
            if link_type not in self.link_types:
                self.link_types.append(link_type)

        # Apply lesion rate if provided
        if lesion_rate is not None and lesion_rate > 0:
            link.lesion_link(lesion_rate)

        # ---- Mask the link to keep only the requested unit pairs ----
        W = link.weights
        if W.ndim != 2:
            raise ValueError(
                f"connect_units expects a 2D weight matrix; got {W.shape}. "
                "Use a full/random/kaiming link, not one-to-one."
            )

        n_out, n_in = W.shape

        # Validate indices
        bad_out = [i for i in outgoing_unit_idx if i < 0 or i >= n_out]
        bad_in  = [j for j in incoming_unit_idx if j < 0 or j >= n_in]
        if bad_out:
            raise IndexError(f"Outgoing unit index/indices out of range: {bad_out} (n_out={n_out})")
        if bad_in:
            raise IndexError(f"Incoming unit index/indices out of range: {bad_in} (n_in={n_in})")

        # Ensure the link has a connection_mask
        if not hasattr(link, "connection_mask") or link.connection_mask is None:
            link.connection_mask = af.ones_like(W)

        # Start from all zeros (overwrite previous mask)
        link.connection_mask[:, :] = 0.0

        # Enable specified pairs
        for oi in outgoing_unit_idx:
            for ij in incoming_unit_idx:
                link.connection_mask[oi, ij] = 1.0

        # Optional reverse direction (mirror semantics; swap groups and index lists)
        if bidirectional:
            self.connect_units(incoming_group, incoming_unit_idx,
                            outgoing_group, outgoing_unit_idx,
                            initialization=initialization,
                            rand_mean=rand_mean,
                            rand_range=rand_range,
                            proj_type=proj_type,
                            link_type=link_type,
                            lesion_rate=lesion_rate,
                            dropout_rate=dropout_rate,
                            perma_lesion_rate=perma_lesion_rate,
                            bidirectional=False)

        return link

    def connect_group_to_unit(self,
                          outgoing_group: str,
                          incoming_group: str,
                          target_unit_idx: list,        # indices within incoming group
                          initialization: str = "uniform",
                          rand_mean: float = 0.0,
                          rand_range: float | None = None,
                          proj_type: str = "full",
                          link_type=None,
                          lesion_rate: float | None = None,
                          dropout_rate: float | None = None,
                          perma_lesion_rate: float | None = None,
                          bidirectional: bool = False):
        """
        Connect *all* units in `outgoing_group` to the specified unit(s) in `incoming_group`.
        Implemented by calling `connect_units(...)` with all outgoing indices.
        """

        # Resolve groups to get counts
        g_out, g_in = self._group_pair_from_names(outgoing_group, incoming_group)
        all_out_idx = list(range(g_out.num_units))

        return self.connect_units(outgoing_group, all_out_idx,
                                incoming_group, target_unit_idx,
                                initialization=initialization,
                                rand_mean=rand_mean,
                                rand_range=rand_range,
                                proj_type=proj_type,
                                link_type=link_type,
                                lesion_rate=lesion_rate,
                                dropout_rate=dropout_rate,
                                perma_lesion_rate=perma_lesion_rate,
                                bidirectional=bidirectional)

    def disconnect_groups(self, group1: str, group2: str, link_type: str | None = None) -> int:
        """
        Deletes links of a specified type between two groups.

        If `link_type` is None, removes *all* links from `group1` to `group2`,
        including Elman projections.

        Returns:
            int: number of links removed
        """
        # Resolve group objects by name (uses existing helper)
        g_out, g_in = self._group_pair_from_names(group1, group2)

        removed = 0
        # Work on a copy since we’ll mutate the lists
        candidate_links = [L for L in list(g_out.outgoing_links)
                        if L.incoming_group is g_in and (link_type is None or L.link_type == link_type)]

        for link in candidate_links:
            # Remove from adjacency lists
            if link in g_out.outgoing_links:
                g_out.outgoing_links.remove(link)
            if link in g_in.incoming_links:
                g_in.incoming_links.remove(link)

            # Remove from global bookkeeping
            lt = getattr(link, "link_type", None)
            if lt in self.links_dict:
                try:
                    self.links_dict[lt].remove(link)
                except ValueError:
                    pass
                # Tidy up empty bins and link_types list
                if not self.links_dict[lt]:
                    del self.links_dict[lt]
                    if lt in self.link_types:
                        self.link_types.remove(lt)
            removed += 1

        if removed == 0:
            if link_type:
                print(f"No links of type '{link_type}' found from '{g_out.name}' to '{g_in.name}'.")
            else:
                print(f"No links found from '{g_out.name}' to '{g_in.name}'.")
        else:
            if link_type:
                print(f"Removed {removed} link(s) of type '{link_type}' from '{g_out.name}' to '{g_in.name}'.")
            else:
                print(f"Removed {removed} link(s) from '{g_out.name}' to '{g_in.name}'.")

        return removed
    
    def disconnect_units(self,
                     outgoing_group: str,
                     outgoing_unit_idx: list,     # indices within outgoing group
                     incoming_group: str,
                     incoming_unit_idx: list,     # indices within incoming group
                     link_type: str | None = None) -> int:
        """
        Deletes links of a specified type between two *units* by clearing the corresponding
        entries in the connection mask. If no `link_type` is given, applies to all link types.
        If a link's mask becomes entirely zero after modification, the link is removed.

        Returns:
            int: number of links fully removed (recycled) due to empty masks.
        """
        # Resolve groups
        g_out, g_in = self._group_pair_from_names(outgoing_group, incoming_group)

        # Gather candidate links from g_out -> g_in (filter by type if provided)
        candidates = [
            L for L in list(getattr(g_out, "outgoing_links", []))
            if getattr(L, "incoming_group", None) is g_in
            and (link_type is None or getattr(L, "link_type", None) == link_type)
        ]

        if not candidates:
            if link_type:
                print(f"No links of type '{link_type}' found from '{g_out.name}' to '{g_in.name}'.")
            else:
                print(f"No links found from '{g_out.name}' to '{g_in.name}'.")
            return 0

        recycled = 0
        pairs_cleared_total = 0

        for link in candidates:
            W = link.weights
            if W.ndim != 2:
                raise ValueError(
                    f"disconnect_units expects a 2D weight matrix; got {W.shape}."
                )

            n_out, n_in = W.shape

            # Validate indices
            bad_out = [i for i in outgoing_unit_idx if i < 0 or i >= n_out]
            bad_in  = [j for j in incoming_unit_idx if j < 0 or j >= n_in]
            if bad_out:
                raise IndexError(f"Outgoing unit index/indices out of range: {bad_out} (n_out={n_out})")
            if bad_in:
                raise IndexError(f"Incoming unit index/indices out of range: {bad_in} (n_in={n_in})")

            # Ensure the link has a mask; if not, treat as fully connected and create one
            if not hasattr(link, "connection_mask") or link.connection_mask is None:
                link.connection_mask = af.ones_like(W)

            mask = link.connection_mask

            # Clear specified pairs
            cleared = 0
            for oi in outgoing_unit_idx:
                for ij in incoming_unit_idx:
                    if mask[oi, ij] != 0.0:
                        mask[oi, ij] = 0.0
                        cleared += 1
            pairs_cleared_total += cleared

            # If mask is now all zeros, recycle/remove this link
            if af.count_nonzero(mask) == 0:
                # Detach from adjacency lists
                if link in g_out.outgoing_links:
                    g_out.outgoing_links.remove(link)
                if link in g_in.incoming_links:
                    g_in.incoming_links.remove(link)

                # Remove from global bookkeeping
                lt = getattr(link, "link_type", None)
                if lt in self.links_dict:
                    try:
                        self.links_dict[lt].remove(link)
                    except ValueError:
                        pass
                    if not self.links_dict[lt]:
                        del self.links_dict[lt]
                        if lt in self.link_types:
                            self.link_types.remove(lt)

                recycled += 1

        # Feedback
        if pairs_cleared_total == 0 and recycled == 0:
            print("Nothing to disconnect for the specified indices.")
        else:
            msg = (f"Disconnected {pairs_cleared_total} pair(s) "
                f"from '{g_out.name}' -> '{g_in.name}'.")
            if link_type:
                msg += f" [type='{link_type}']"
            if recycled:
                msg += f" Recycled {recycled} empty link(s)."
            print(msg)
        return pairs_cleared_total
    
    def disconnect_group_units(self,
                          outgoing_group: str,
                          incoming_group: str,
                          incoming_unit_idx: list,
                          link_type: str | None = None) -> int:
        """
        disconnectGroupUnit - deletes links from a *group* to one or more *units*.

        This calls `disconnect_units` using all units in the outgoing group and
        the provided incoming unit indices.

        Example:
            disconnect_group_unit("input", "hidden", [3], link_type="special")
            → disconnects all links of type "special" from input[:] to hidden[3].

            disconnect_group_unit("input", "hidden", [2, 3])
            → disconnects all links (all types by default) from input[:] to hidden[2] and hidden[3].
        """
        # Resolve group names
        g_out, g_in = self._group_pair_from_names(outgoing_group, incoming_group)

        # All source indices from the outgoing group
        outgoing_unit_idx = list(range(g_out.num_units))

        # Delegate to disconnect_units (reuse full logic)
        return self.disconnect_units(
            outgoing_group,
            outgoing_unit_idx,
            incoming_group,
            incoming_unit_idx,
            link_type=link_type
        )
        
    def delete_links(self, link_type: str | None = None) -> int:
        """
        deleteLinks - deletes all links of a specified type.
        If `link_type` is None, deletes ALL links in the network.

        Returns:
            int: number of link objects removed.
        """
        removed = 0
        # Collect target links
        if link_type is None:
            targets = []
            for lst in list(self.links_dict.values()):
                targets.extend(lst)
        else:
            targets = list(self.links_dict.get(link_type, []))
        if not targets:
            if link_type is None:
                print("No links to delete.")
            else:
                print(f"No links of type '{link_type}' found.")
            return 0

        # Detach each link from its groups' adjacency lists
        for link in targets:
            g_out = getattr(link, "outgoing_group", None)
            g_in  = getattr(link, "incoming_group", None)
            if g_out is not None and hasattr(g_out, "outgoing_links"):
                try:
                    g_out.outgoing_links.remove(link)
                except ValueError:
                    pass
            if g_in is not None and hasattr(g_in, "incoming_links"):
                try:
                    g_in.incoming_links.remove(link)
                except ValueError:
                    pass
            removed += 1
        if link_type is None:
            self.links_dict.clear()
            self.link_types.clear()
            print(f"Deleted {removed} link(s) (all types).")
        else:
            if link_type in self.links_dict:
                del self.links_dict[link_type]
            if link_type in self.link_types:
                try:
                    self.link_types.remove(link_type)
                except ValueError:
                    pass
            print(f"Deleted {removed} link(s) of type '{link_type}'.")
            
    def delete_unit_inputs(self,
                       incoming_group: str,
                       incoming_unit_idx: list[int],
                       link_type: str | None = None) -> int:
        """
        deleteUnitInputs — remove *all* inputs into the specified units of `incoming_group`.

        This is a bulk "fan-in" delete: for each unit index in `incoming_unit_idx`,
        disconnect all presynaptic groups that currently project into `incoming_group`.
        If `link_type` is provided, only connections of that type are removed.

        Args:
            incoming_group: name of the destination group
            incoming_unit_idx: list of destination unit indices within `incoming_group`
            link_type: optional link type filter

        Returns:
            int: total number of connections disabled across all affected links.
        """
        
        _, g_in = self._group_pair_from_names(incoming_group, incoming_group)
        if not incoming_unit_idx:
            return 0
        # Validate indices against the group size (if available)
        if hasattr(g_in, "num_units"):
            bad = [j for j in incoming_unit_idx if j < 0 or j >= g_in.num_units]
            if bad:
                raise IndexError(
                    f"Unit index/indices out of range for group '{incoming_group}': "
                    f"{bad} (num_units={g_in.num_units})"
                )
        # Collect distinct groups that actually connect into this group
        presyn_groups = []
        seen = set()
        for link in getattr(g_in, "incoming_links", []) or []:
            g_out = getattr(link, "outgoing_group", None)
            if g_out is None:
                continue
            key = id(g_out)
            if key not in seen:
                seen.add(key)
                presyn_groups.append(g_out)

        if not presyn_groups:
            # Nothing feeds this group; nothing to remove.
            print(f"No incoming links found for group '{incoming_group}'.")
            return 0

        # For each group, disconnect its projection to the target unit(s)
        total_disabled = 0
        for g_out in presyn_groups:
            total_disabled += self.disconnect_group_units(
                outgoing_group=g_out.name,
                incoming_group=incoming_group,
                incoming_unit_idx=incoming_unit_idx,
                link_type=link_type,
            )
        if total_disabled == 0:
            if link_type:
                print(f"No inputs of type '{link_type}' were removed for units {incoming_unit_idx} in '{incoming_group}'.")
            else:
                print(f"No inputs were removed for units {incoming_unit_idx} in '{incoming_group}'.")
        else:
            if link_type:
                print(f"Removed {total_disabled} input connection(s) of type '{link_type}' "
                    f"into '{incoming_group}' for units {incoming_unit_idx}.")
            else:
                print(f"Removed {total_disabled} input connection(s) into '{incoming_group}' for units {incoming_unit_idx}'.")

        return total_disabled
    
    def delete_group_inputs(self,
                        incoming_group: str,
                        link_type: str | None = None) -> int:
        """
        deleteGroupInputs — deletes *all* inputs into every unit of `incoming_group`.
        If `link_type` is provided, only inputs of that type are removed.

        This is the group-wide version of `delete_unit_inputs(...)`.
        """
        # Get the group object and its size
        _, g_in = self._group_pair_from_names(incoming_group, incoming_group)

        # All unit indices in the destination group
        all_units = list(range(g_in.num_units))

        # Reuse the per-unit bulk deleter
        return self.delete_unit_inputs(
            incoming_group=incoming_group,
            incoming_unit_idx=all_units,
            link_type=link_type,
        )
        
    def delete_unit_outputs(self,
                        outgoing_group: str,
                        outgoing_unit_idx: list[int],
                        link_type: str | None = None) -> int:
        """
        deleteUnitOutputs — deletes *all outputs* of the specified type from units
        in `outgoing_group`.

        This removes all outgoing connections from the given unit indices to every
        postsynaptic group that they project to. If `link_type` is provided, only
        links of that type are affected.
        """

        g_out, _ = self._group_pair_from_names(outgoing_group, outgoing_group)
        if not outgoing_unit_idx:
            return 0
        # Validate indices
        if hasattr(g_out, "num_units"):
            bad = [i for i in outgoing_unit_idx if i < 0 or i >= g_out.num_units]
            if bad:
                raise IndexError(
                    f"Outgoing unit index/indices out of range for group '{outgoing_group}': "
                    f"{bad} (num_units={g_out.num_units})"
                )
        total_disabled = 0
        found_any = False

        for link in list(getattr(g_out, "outgoing_links", []) or []):
            if link_type is not None and getattr(link, "link_type", None) != link_type:
                continue
            W = link.weights
            if W.ndim != 2:
                continue
            if getattr(link, "connection_mask", None) is None:
                link.connection_mask = af.ones_like(W, dtype=af.float32)

            for i in outgoing_unit_idx:
                before = int(link.connection_mask[i, :].sum())
                if before:
                    link.connection_mask[i, :] = 0.0
                    total_disabled += before
            found_any = True

        if not found_any:
            if link_type:
                print(f"No links of type '{link_type}' found from '{g_out.name}'.")
            else:
                print(f"No outgoing links found from '{g_out.name}'.")
        else:
            if total_disabled == 0:
                print(f"No active outputs to disable for units {outgoing_unit_idx} in '{g_out.name}'.")
            else:
                if link_type:
                    print(f"Removed {total_disabled} output connection(s) of type '{link_type}' "
                        f"from '{g_out.name}' for units {outgoing_unit_idx}.")
                else:
                    print(f"Removed {total_disabled} output connection(s) from '{g_out.name}' "
                        f"for units {outgoing_unit_idx}.")

        return total_disabled
    
    def delete_group_outputs(self,
                         outgoing_group: str,
                         link_type: str | None = None) -> int:
        """
        deleteGroupOutputs — deletes *all* outgoing connections* from every unit in `outgoing_group`.
        If `link_type` is provided, only connections of that type are removed.

        This is the group-level version of `delete_unit_outputs`.
        """
        g_out, _ = self._group_pair_from_names(outgoing_group, outgoing_group)

        # All unit indices in the source group
        all_units = list(range(g_out.num_units))

        # Delegate to the per-unit deleter
        return self.delete_unit_outputs(
            outgoing_group=outgoing_group,
            outgoing_unit_idx=all_units,
            link_type=link_type,
        )
    
    def order_groups(self, group_order):
        """
        Orders the groups in the network based on the given order.
        
        Args:
            group_order (List[str]): the desired order of the groups in the network.
                Must include all groups currently in the network.
        """
        ordered_groups = []
        for i, name in enumerate(group_order):
            if i == 0 and name == "bias":   # special case for first bias group
                for group in self.groups:
                    if group.name == "bias":
                        ordered_groups.append(group)
                continue
            found = False
            for group in self.groups:
                if group.name == name:
                    ordered_groups.append(group)
                    found = True
                    break
            if not found:
                raise Exception("Group name {} not found in network".format(name))
        if len(ordered_groups) != len(self.groups):
            raise Exception("Group order does not include all groups in the network")
        self.groups = ordered_groups
    
    def delete_group(self, group_names=[], delete_all=False):
        """
        Deletes groups from the network.

        Args:
            group_names (list, optional): List of group names to delete. Defaults to [].
            delete_all (bool, optional): If True, deletes all groups except bias. Defaults to False.
        """
        if delete_all:
            for group in self.groups[:]:
                if group.group_type != "bias":
                    self.groups.remove(group)
                    self.num_groups -= 1
                    self.num_units -= group.num_units
                    if group in self.input_groups:
                        self.input_groups.remove(group)
                    if group in self.output_groups:
                        self.output_groups.remove(group)
            return
        for name in group_names:
            group = self.get_group_by_name(name)
            if group.group_type == "bias":
                print("Cannot delete bias group")
                continue
            self.groups.remove(group)
            self.num_groups -= 1
            self.num_units -= group.num_units
            if group in self.input_groups:
                self.input_groups.remove(group)
            if group in self.output_groups:
                self.output_groups.remove(group)

    def get_group_type(self, group_name=None):
        """
        Print information about a group's type, input type, and output type.

        If no group name is provided, print all available base types, input types,
        and output types.

        Args:
            group_name (str, optional): Name of the group to query. Defaults to None.
        """
        if not group_name:
            print("Base Types: ")
            for group_type in Group.total_group_types:
                print("   -" + group_type)
            print("Input Types: ")
            for input_type in Group.input_types.keys():
                print("   -" + input_type)
            print("Output Types: ")
            for output_type in Group.activations.keys():
                print("   -" + output_type)
            return
        group = self.get_group_by_name(group_name)
        print("Group Name: {}".format(group.name))
        print("Group Type: {}".format(group.group_type))
        print("Input Type: ")
        for inp in group.input_transforms:
            print("   -" + inp.name)
        print("Output Type: ")
        for out in group.output_transforms:
            print("   -" + out.name)
    
    def change_group_type(self, group_name, new_input_transforms=[], new_output_transforms=[]):
        """
        Change the type of a group by updating its input and output transforms.

        Args:
            group_name (str): The name of the group to change.
            new_input_transforms (list, optional): List of new input transforms. Defaults to [].
            new_output_transforms (list, optional): List of new output transforms. Defaults to [].
        """
        group = self.get_group_by_name(group_name)
        for transform in new_input_transforms:
            group.input_transforms.append(Group.input_types[transform](group))
        for transform in new_output_transforms:
            group.output_transforms.append(Group.activations[transform](group))

    def copy_unit_values(
        self,
        src_group_name: str,
        dst_group_name: str = None,
        *,
        src_field: str = "outputs",
        dst_field: str = None,
        update_cache: bool = True,
        require_same_units: bool = True,
    ) -> None:
        """
        Copy unit values from one group's field to another group's field, or within the same group.

        Args:
            src_group_name (str): Name of the source group.
            dst_group_name (str, optional): Name of the destination group.
                If None, defaults to ``src_group_name``.
            src_field (str): Source field name. One of:
                ``{"inputs", "outputs", "targets", "input_derivs",
                "output_derivs", "external_input"}``.
            dst_field (str, optional): Destination field. Defaults to ``src_field``.
            update_cache (bool): If True and writing to outputs, refresh cached outputs.
            require_same_units (bool): If True, require that both groups have
                the same number of units.
        """
        # map field names → Group attributes
        attr_map = {
            "inputs": "input_matrix",
            "outputs": "output_matrix",
            "targets": "target",
            "input_derivs": "input_derivs",
            "output_derivs": "output_derivs",
            "external_input": "external_input",
        }

        if src_field not in attr_map:
            raise ValueError(f"Invalid src_field '{src_field}'. Valid: {list(attr_map)}")

        if dst_field is None:
            dst_field = src_field
        if dst_field not in attr_map:
            raise ValueError(f"Invalid dst_field '{dst_field}'. Valid: {list(attr_map)}")

        src_group = self.get_group_by_name(src_group_name)
        dst_group = self.get_group_by_name(dst_group_name) if dst_group_name else src_group

        # sanity: check units
        if require_same_units and src_group.num_units != dst_group.num_units:
            raise ValueError(
                f"Unit mismatch: '{src_group_name}' has {src_group.num_units}, "
                f"'{dst_group.name}' has {dst_group.num_units}"
            )

        src_arr = getattr(src_group, attr_map[src_field])
        if not hasattr(dst_group, attr_map[dst_field]):
            raise ValueError(f"Destination field '{dst_field}' not found in group '{dst_group.name}'")

        # instantaneous copy (not reference sharing)
        setattr(dst_group, attr_map[dst_field], copy.copy(src_arr))

        # ensure caches stay valid for outputs
        if update_cache and attr_map[dst_field] == "output_matrix":
            dst_group.cache_outputs()
            
    # network.py  (inside class Network)
    
    def print_unit_values(
        self,
        filename: str,
        group_names="*",
        append: bool = False,
    ) -> int:
        """
        Print basic unit values for the given groups into a file.
        Creates file if not exists. Supports .gz and .bz2.

        Format:
          group_name unit_index field_name value(s)

        Returns: number of lines written
        """
        fields = {
            "inputs": "input_matrix",
            "outputs": "output_matrix",
            "targets": "target",
            "input_derivs": "input_derivs",
            "output_derivs": "output_derivs",
            "external_input": "external_input",
        }

        groups = self._resolve_groups(group_names)
        mode = "a" if append else "w"

        lines = 0
        with self._open_text(filename, mode) as f:
            for g in groups:
                for fname, attr in fields.items():
                    if not hasattr(g, attr):
                        continue
                    arr = getattr(g, attr)
                    if arr is None:
                        continue

                    arr = af.asarray(arr)

                    # if 1D, treat as one value per unit
                    if arr.ndim == 1:
                        for u in range(min(g.num_units, arr.shape[0])):
                            # skip lesioned units
                            if g.lesion_mask is not None and g.lesion_mask[u] == 0:
                                continue
                            f.write(f"{g.name} {u} {fname} {arr[u]}\n")
                            lines += 1
                    else:
                        # assume first axis is units
                        for u in range(min(g.num_units, arr.shape[0])):
                            # skip lesioned units
                            if g.lesion_mask is not None and g.lesion_mask[u] == 0:
                                continue
                            vals = " ".join(str(x) for x in af.ravel(arr[u]))
                            f.write(f"{g.name} {u} {fname} {vals}\n")
                            lines += 1
        return lines


    def polarity(self, action: str, group_names="*"):
        """
        Polarity command: 'reset' | 'update' | 'report'

        - reset:   zero out polarity_sum / polarity_num
        - update:  accumulate current polarity from outputs into sum/num
        - report:  return {group_name: avg_polarity_since_last_reset_or_update}
        """
        action = action.lower()
        groups = self._resolve_groups(group_names)

        if action.startswith("reset"):
            for g in groups:
                # lazily attach the fields if they don't exist yet
                if not hasattr(g, "polarity_sum"):
                    g.polarity_sum = 0.0
                if not hasattr(g, "polarity_num"):
                    g.polarity_num = 0
                g.polarity_sum = 0.0
                g.polarity_num = 0

        elif action.startswith("update"):
            # compute and accumulate polarity for each unit's current output
            # Formula (x in [0,1]):
            # p(x) = x*log2(x) + (1-x)*log2(1-x) + 1
            # If x<=0 or x>=1, treat as 1.0 (as in the C reference).
            ln2 = af.log(2.0)

            for g in groups:
                # ensure accumulators exist
                if not hasattr(g, "polarity_sum"):
                    g.polarity_sum = 0.0
                if not hasattr(g, "polarity_num"):
                    g.polarity_num = 0

                # normalize outputs to [0,1] using per-group range
                scale = 1.0 / (g.maxOutput - g.minOutput)
                y = g.output_matrix  # shape (num_units, ...) — treat as flat vector
                x = (y - g.minOutput) * scale
                x = af.ravel(x)

                # piecewise as in C code
                mask = (x <= 0.0) | (x >= 1.0)
                inner = (x * af.log(x) + (1.0 - x) * af.log(1.0 - x)) / ln2 + 1.0
                d = af.where(mask, 1.0, inner)

                g.polarity_sum += float(af.sum(d))
                g.polarity_num += int(d.size)

        elif action.startswith("report"):
            report = {}
            for g in groups:
                s = getattr(g, "polarity_sum", 0.0)
                n = getattr(g, "polarity_num", 0)
                report[g.name] = (s / n) if n > 0 else 0.0
            print(report)

        else:
            raise ValueError("polarity(action, ...): action must be one of {'reset','update','report'}")
    
    def add_link_type(self, new_type):
        if new_type not in self.link_types:
            self.link_types.append(new_type)
        return self.link_types

    def delete_link_type(self, target_type):
        if target_type in self.link_types:
            self.link_types.remove(target_type)
        return self.link_types
    
    def set_link_values(
        self,
        parameter: str,
        value,
        *,
        group_names="*",
        link_type: str | None = None,
    ) -> int:
        """
        Set a parameter on all *incoming* links into the selected groups.
        Filters by link_type if provided.

        Supported parameters:
          learningRate -> link.link_learning_rate
          momentum     -> link.link_params.PAR_O_momentum
          weightDecay  -> link.link_params.PAR_O_weightDecay
          randMean     -> link.link_params.PAR_O_randMean
          randRange    -> link.link_params.PAR_O_randRange
          min          -> link.min_weights
          max          -> link.max_weights

        Returns: number of links updated.
        """
        param_map = {
            "learningrate": ("direct", "link_learning_rate"),
            "momentum":     ("param",  "PAR_O_momentum"),
            "weightdecay":  ("param",  "PAR_O_weightDecay"),
            "randmean":     ("param",  "PAR_O_randMean"),
            "randrange":    ("param",  "PAR_O_randRange"),
            "min":          ("direct", "min_weights"),
            "max":          ("direct", "max_weights"),
        }
        
        key = parameter.replace("_", "").lower()
        if key not in param_map:
            raise ValueError(f"Unsupported parameter '{parameter}'. Choose from {list(param_map)}")
        
        where, attr = param_map[key]
        set_val = af.nan if (isinstance(value, str) and value.strip() == "-") else float(value)

        groups = self._resolve_groups(group_names)
        updated = 0
        
        for g in groups:
            incoming = getattr(g, "incoming_links", None)
            if not incoming:
                continue

            for L in incoming:
                # optional filter by link_type (match your Link schema)
                if link_type is not None:
                    lt = getattr(L, "link_type", None)  # e.g., "uniform", "elman", "bias", etc.
                    if lt != link_type:
                        continue

                if where == "direct":
                    if not hasattr(L, attr):
                        raise AttributeError(f"Link missing attribute '{attr}'")
                    setattr(L, attr, set_val)
                else:  # parameter on link.link_params
                    params = getattr(L, "link_params", None)
                    if params is None or not hasattr(params, attr):
                        raise AttributeError(f"Link missing link_params.{attr}")
                    setattr(params, attr, set_val)

                updated += 1

        return updated
    
    def print_link_values(
        self,
        filename: str,
        senders="*",
        receivers="*",
        *,
        link_type: str | None = None,
        append: bool = False,
    ) -> int:
        """
        Write info about links (weights) from sender groups -> receiver groups.

        Args:
            filename: output file (.gz/.bz2 supported)
            senders: sender group(s), "*" for all
            receivers: receiver group(s), "*" for all
            link_type: optional filter (e.g. "uniform", "elman", "bias")
            append: if True, append to file, else overwrite

        Format per line:
            sender_group sender_unit -> recv_group recv_unit [type] weight

        Returns:
            number of links written
        """
        send_groups = self._resolve_groups(senders)
        recv_groups = self._resolve_groups(receivers)
        mode = "a" if append else "w"

        lines = 0
        with self._open_text(filename, mode) as f:
            for rg in recv_groups:
                for link in rg.incoming_links:
                    if link_type is not None and getattr(link, "link_type", None) != link_type:
                        continue
                    sg = link.outgoing_group
                    if sg not in send_groups:
                        continue

                    weights = getattr(link, "weights", None)

                    f.write(
                        f"{sg.name} {sg.num_units} -> {rg.name} {rg.num_units} "
                        f"[{link.link_type}] {weights}\n"
                    )
                    lines += 1
        return lines
        
    def forward(self, tick):
        """
        Computes the forward pass of the network, iterating through the list of groups

        Args:
            tick (int): passing in the length of time the example will occur for
        """
        group_outputs = []

        # iterate through all of the groups and compute the forward pass for each
        for group in self.groups[:]:
            if (group.group_type != "bias"):
                group_outputs += [group.forward(tick)]

        return group_outputs

    def backward(self):
        """
        Computes the backward pass of the network, iterating through the list of groups in reverse order
        
        Returns:
            group_outputs (List): the output of each group in the network
        """
        group_outputs = []

        for group in reversed(self.groups):
            group_outputs += [group.backward()]

        return group_outputs

    def update_weights(self, report_request=False):
        """
        Apply one optimizer weight update using accumulated link derivatives.

        Delegates to ``self.optimizer.update_weights``. For assigning a concrete
        weight matrix to a single connection, use ``Link.update_weight(new_weight)``.

        Args:
            report_request (bool): If True, optimizers may gather extra report statistics.
        """
        self.optimizer.update_weights(report_request=report_request)

    def load_input(self, input_matrix):
        """
        Loads the data from example sets into the input groups

        Args:
            input_matrix (List): The matrix of data inputted into the network
        """
        for i in range(len(self.input_groups)):
            try:
                self.input_groups[i].set_external_input(input_matrix[i])
            except IndexError:
                print("error: number of input groups {} does not match examples {}".format(
                    len(self.input_groups), 
                    len(input_matrix))
                      )

    def load_target(self, target_matrix):
        """Loads target values into the output groups."""
        for group, target in zip(self.output_groups, target_matrix):
            group.set_target(target)

    def load_event(self, event):
        """Loads an event's inputs and targets into the network."""
        self.load_input(event.input_group)
        self.load_target(event.target_group)

    def load_example_set(self, file_name: str, name="example", proc=False, 
                         default_input=0, active_input=1, default_target=0, 
                         active_target=1, num_examples_loaded=None,
                         def_s_max_time=example_params.DEF_S_MAXTIME, 
                         def_s_min_time=example_params.DEF_S_MAXTIME,
                         training=True, testing=False, sort_mode="ORDERED"):
        """
        Loads in an example set into the network object

        Args:
            file_name (str): name of the file containing the example set
            name (str): name of the example set loading in
            proc (boolean): whether to process the data
            default_input (int): default value for input
            active_input (int): active value for input
            default_target (int): default value for target
            active_target (int): active value for target
            num_examples_loaded (int): number of examples loaded
            def_s_max_time (int): default maximum time
            def_s_min_time (int): default minimum time
            training (boolean): whether the example set is for training
            testing (boolean): whether the example set is for testing
        """
        self.record_action("load_example_set", 
                           file_name=file_name, 
                           name=name, 
                           proc=proc, 
                           default_input=default_input, 
                           active_input=active_input, 
                           default_target=default_target, 
                           active_target=active_target, 
                           num_examples_loaded=num_examples_loaded, 
                           def_s_max_time=def_s_max_time, 
                           def_s_min_time=def_s_min_time, 
                           training=training, 
                           testing=testing)
        if '.' not in file_name:
            file_name = "./lens_example_input/{0}.ex".format(file_name)
        new_example_set = ExampleSet.initialize_example_set(self, proc, name, 
                                                            file_name, 
                                                            self.input_groups,
                                                            self.output_groups, 
                                                            default_input, 
                                                            active_input,
                                                            default_target, active_target,
                                                            def_s_max_time=def_s_max_time,
                                                            def_s_min_time=def_s_min_time,
                                                            num_loaded=num_examples_loaded)
        new_example_set.set_sort_mode(sort_mode)
        if not new_example_set:
            return None
        else:
            if training:
                self.training_sets.append(new_example_set)
                self.loaded_example_sets[new_example_set.name] = new_example_set
            if testing:
                self.testing_sets.append(new_example_set)
                self.loaded_example_sets[new_example_set.name] = new_example_set

            if self.training_set is None:
                self.training_set = new_example_set

            if self.testing_set is None or self.testing_set == self.training_set:
                self.testing_set = new_example_set
            return True

    def move_examples(
        self,
        src_name: str,
        dst_name: str,
        first_example: int | None = None,
        num_examples: int | None = None,
        proportion: float | None = None,
        copy_examples: bool = False,
    ):
        """
        Move (or copy) examples from one ExampleSet to another, similar to Lens `moveExamples`.

        Args:
            src_name (str): name of source example set.
            dst_name (str): name of destination example set (must already exist for now).
            first_example (int | None): index of first example to move; 0-based.
            num_examples (int | None): how many examples to move starting from first_example.
            proportion (float | None): if given, ignore first_example/num_examples and
                move this proportion (0–1) of examples at random.
            copy_examples (bool): if True, copy instead of move.
        """
        # --- 1. Look up sets ---
        if src_name not in self.loaded_example_sets:
            raise ValueError(f'source example set "{src_name}" is not loaded.')
        if dst_name not in self.loaded_example_sets:
            raise ValueError(f'destination example set "{dst_name}" is not loaded.')

        src_set = self.loaded_example_sets[src_name]
        dst_set = self.loaded_example_sets[dst_name]

        if src_set.num_examples == 0:
            return  # nothing to do

        n_src = src_set.num_examples

        # --- 2. Decide which indices to move ---
        if proportion is not None:
            if not (0.0 < proportion <= 1.0):
                raise ValueError("proportion must be in (0, 1].")
            k = max(1, int(round(proportion * n_src)))
            indices = sorted(random.sample(range(n_src), k))
        elif first_example is not None:
            if first_example < 0 or first_example >= n_src:
                raise IndexError(
                    f"first_example {first_example} out of range (0–{n_src - 1})"
                )
            if num_examples is None:
                num_examples = 1
            end = min(first_example + num_examples, n_src)
            indices = list(range(first_example, end))
        else:
            # move all
            indices = list(range(n_src))

        if not indices:
            return

        # --- 3. Collect examples in the chosen order ---
        src_examples = src_set.example
        selected_examples = [src_examples[i] for i in indices]

        # --- 4. Copy or move ---
        if copy_examples:
            # COPY: source remains unchanged; deep copy into dst_set
            for ex in selected_examples:
                memo = {id(ex.network): ex.network}
                new_ex = copy.deepcopy(ex, memo)
                new_ex.set = dst_set
                dst_set.register_example(new_ex, new=True)
            dst_set.num_examples = len(dst_set.example)
        else:
            # MOVE: remove from src, append to dst
            # Build remaining list for src, preserving order
            remaining = [ex for i, ex in enumerate(src_examples) if i not in set(indices)]

            # Reset src examples & relink via register_example
            src_set.example = []
            src_set.first_example = None
            src_set.last_example = None
            for ex in remaining:
                src_set.register_example(ex, new=True)
            src_set.num_examples = len(src_set.example)

            # Append moved examples to dst
            for ex in selected_examples:
                ex.set = dst_set
                dst_set.register_example(ex, new=True)
            dst_set.num_examples = len(dst_set.example)

        # --- 5. Rebuild iterators so ORDERED / PERMUTED / etc still work ---
        src_set.example_iterator = ExampleIterator.init_example_iterator(src_set)
        dst_set.example_iterator = ExampleIterator.init_example_iterator(dst_set)


    def use_training_set(self, example_set_name: str | None = None):
        """
        Sets or lists the current training set for this network.

        Args:
            example_set_name (str | None): Name of the example set to use.
                - If None: return list of available example sets.
                - If empty string "": clear the current training set.
        """
        # No argument: list all available example sets
        if example_set_name is None:
            return list(self.loaded_example_sets.keys())

        # Empty string or {} equivalent → clear training set
        if example_set_name.strip() == "":
            self.training_set = None
            return None

        # Find and set
        if example_set_name not in self.loaded_example_sets:
            raise ValueError(f'Example set "{example_set_name}" does not exist.')

        self.training_set = self.loaded_example_sets[example_set_name]
        return None
        
    def use_testing_set(self, example_set_name: str | None = None):
        """
        Sets or lists the current testing set for this network.

        Args:
            example_set_name (str | None): Name of the example set to use.
                - If None: return list of available example sets.
                - If empty string "": clear the current testing set.
        """
        if example_set_name is None:
            return list(self.example_sets.keys())

        if example_set_name.strip() == "":
            self.testing_set = None
            return None

        if example_set_name not in self.loaded_example_sets:
            raise ValueError(f'Example set "{example_set_name}" does not exist.')

        self.testing_set = self.loaded_example_sets[example_set_name]
        return None
    
    def delete_example_sets(self, set_names: list[str] | str):
        """
        Deletes example sets by name, or all if '*' is provided.

        Args:
            set_names (list[str] | str):
                List of example set names, or "*" to delete all.
        """
        if isinstance(set_names, str):
            if set_names.strip() == "*":
                # Delete all
                self.training_sets.clear()
                self.testing_sets.clear()
                self.training_set = None
                self.testing_set = None
                self.loaded_example_sets.clear()
                return
            else:
                set_names = set_names.strip().split()

        # Filter out matching names
        self.training_sets = [
            s for s in self.training_sets if s.name not in set_names
        ]
        self.testing_sets = [
            s for s in self.testing_sets if s.name not in set_names
        ]
        
        for name in set_names:
            self.loaded_example_sets.pop(name, None)

        # Clear current pointers if deleted
        if self.training_set and self.training_set.name in set_names:
            self.training_set = None
        if self.testing_set and self.testing_set.name in set_names:
            self.testing_set = None

    def standard_net_train_tick(self, event, tick, example):
        """
        Iterates over the event and runs for the desired time

        Args:
            event (Event): the current event
            tick (int): tick of the network
            example (Example): example of the event
            
        Returns:
            input_result (List): the result of the input
            sum(self.errors) (float): the sum of the errors
            sum(self.unit_cost) (float): the sum of the unit costs
        """
        input_result = []
        if event.pre_proc_name is not None:
            event.pre_proc()

        target = self.output_groups[0].target

        if self.network_type == 'standard':
            self.reset_derivs()
        group_outputs = self.forward(tick) # continuous network, srbptt, and other networks have different self.forward methods.
        output = group_outputs[-1]

        # check if output is greater than threshold. If so, update criterion?
        for i in range(len(output)):
            if abs(output[i] - target[i]) < self.group_criterion_threshold:
                self.group_criterion_reached = True
            else:
                self.group_criterion_reached = False

        group_outputs.append(target)
        input_result.append([s.tolist() for s in group_outputs])

        self.errors, self.error_derivs = self.compute_cost(
                self.output_groups, 
                example.frequency, 
                tick)
        self.unit_cost, self.unit_cost_derivs = self.compute_unit_output_cost(self.output_groups)
        example.example_train_error.append(sum(self.errors))

        # Accumulate errors over the batch
        if self.batch_errors is None:
            self.batch_errors = self.errors
        else:
            self.batch_errors = [i + j for i, j in zip(self.batch_errors, self.errors)]
        # Accumulate unit costs over the batch
        if self.batch_unit_costs is None:
            self.batch_unit_costs = self.unit_cost
        else:
            self.batch_unit_costs = [i + j for i, j in zip(self.batch_unit_costs, self.unit_cost)]

        if self.network_type == 'continuous':
            # here it transfers the derivatives to output groups, the backprop for the groups are then done by net_train_example_back()
            for i in range(len(self.output_groups)):
                self.output_groups[i].output_derivs = self.error_derivs[i] + self.unit_cost_derivs[i]
                if self.parallel_mode:
                    self.output_groups[i].output_derivs_history = self.output_groups[i].output_derivs_history.copy()
                self.output_groups[i].output_derivs_history[tick] = self.error_derivs[i] + self.unit_cost_derivs[i]
        else:
            output_derivs = self.compute_back()
            for group in self.groups:
                af.fill(group.outputderivCache, 0)

        if event.post_proc_name is not None:
            event.post_proc()
        # self.reset_matrices()
        # print("reset matrices")
        return input_result, sum(self.errors), sum(self.unit_cost)

    def reset_derivs(self):
        """
        Resets the derivatives of the network
        """
        for group in self.groups:
            group.clear_derivs()

    def reset_matrices(self):
        """
        Resets the matrices of the network
        """
        for group in self.groups:
            if group.group_type == "elman":
                af.fill(group.output_matrix, 0.5)
                group.reset_input = True

    def standard_net_test_tick(self, event, tick, example):
        """
        Iterates over the event and runs for the desired time to perform forward pass for testing

        Args:
            event (Event): the current event
            tick (int): tick of the network
            
        Returns:
            input_result (List)
        """

        input_result = []
        target = self.output_groups[0].target

        self.reset_derivs()
        group_outputs = self.forward(tick)
        output = group_outputs[-1]

        # check if output is greater than threshold. If so, update criterion?
        for i in range(len(output)):
            if abs(output[i] - target[i]) < self.test_group_criterion_threshold:
                self.test_group_criterion_reached = True
            else:
                self.test_group_criterion_reached = False

        group_outputs.append(target)
        input_result.append([s.tolist() for s in group_outputs])
        self.test_errors, self.test_error_derivs = self.compute_cost(
                self.output_groups, 
                example.frequency, 
                tick
                )
        self.test_unit_cost, self.test_unit_cost_derivs = self.compute_unit_output_cost(self.output_groups)
        example.example_test_error += sum(self.test_errors)

        # Accumulate errors over the batch
        if self.batch_test_errors is None:
            self.batch_test_errors = self.test_errors
        else:
            self.batch_test_errors = [i + j for i, j in zip(self.batch_test_errors, self.test_errors)]
        if sum(self.batch_test_errors) < self.batch_test_errors_threshold:
            self.test_error_criterion = True
            return

        # Accumulate unit costs over the batch
        if self.batch_test_unit_cost is None:
            self.batch_test_unit_cost = self.test_unit_cost
        else:
            self.batch_test_unit_cost = [i + j for i, j in zip(self.batch_test_unit_cost, self.test_unit_cost)]

        return input_result

    def standard_net_train_example(self, example, test=False):
        """
        Iterates over the current example

        Args:
            example (Example): the example to be trained or tested
            test (boolean): whether the example is for testing or training
            
        Returns:
            event_result (List): results from the processed events
            training_errors (List): the training errors
            unit_costs (List): the unit costs
        """
        if self.network_type == 'continuous':
            first_tick = 1
        else:
            first_tick = 0
        ticks_on_event = 0
        event_result = []
        training_errors = []
        unit_costs = []
        target_str = ""

        if example.pre_proc_name is not None:
            example.pre_proc()

        # Reset the outputs and integrators
        # pre load of first event in order to properly reset_output
        event_index = 0
        event = example.event[event_index]
        self.load_event(event)
        for group in self.groups[:]:
            self.reset_outputs(group)
            self.reset_forwardintegrators(group)

        # store outputs and targets for continuous network
        if self.network_type == 'continuous':
            for group in self.groups[:]:
                if group.group_type != "bias":
                    if self.parallel_mode:
                        group.output_history = group.output_history.copy()
                        group.input_history = group.input_history.copy()
                    group.output_history[first_tick-1] = group.output_matrix
                    group.input_history[first_tick-1] = group.input_matrix

        if event.max_time is not None:
            max_time = event.max_time
        elif example.set.max_time is not None:
            max_time = example.set.max_time
        else:
            max_time = self.max_example_time

        for tick in range(first_tick, self.max_ticks):
            if event_index >= len(example.event):
                break

            # Record the target once when this event begins.
            if ticks_on_event == 0:
                for targ in event.target_group:
                    target_str += ' '.join(
                        map(str, af.astype(targ, int).tolist())
                    )
                    target_str += " "

            self.current_tick = tick

            if test:
                event_result += self.standard_net_test_tick(event, tick, example)
                if self.test_error_criterion or self.test_group_criterion_reached:
                    return
            else:
                result, error, unit_cost = self.standard_net_train_tick(event, tick, example)
                event_result += result
                training_errors.append(error)
                unit_costs.append(unit_cost)

            ticks_on_event += 1
            time_on_event = ticks_on_event / self.ticks_per_interval

            # stores external input history
            if test:
                for g in self.groups:
                    if g.group_type == "input":
                        g.external_input_history.append(g.external_input)

            if time_on_event >= max_time or tick >= self.max_ticks - 1:
                # event done
                event_index += 1
                ticks_on_event = 0

                if event_index < len(example.event):
                    event = example.event[event_index]
                    self.load_event(event)

                    # set max_time
                    if event.max_time is not None:
                        max_time = event.max_time
                    elif example.set.max_time is not None:
                        max_time = example.set.max_time
                    else:
                        max_time = self.max_example_time

        self.ticks_on_example = self.current_tick + 1

        self.res = str(example.name) + "|output "
        for outg in self.output_groups:
            self.res += ' '.join(map(str, outg.output_matrix.tolist())) + " "
        self.res += "\n" + str(example.name) + "|target " + target_str + "\n"
        if example.post_proc_name is not None:
            example.post_proc()

        return event_result, training_errors, unit_costs
    
    def do_example(self, example_set_name: str | None = None,
               example_index: int | None = None,
               test: bool = False):
        """
        Runs a single example (training or testing) from a specified or current example set.

        Args:
            example_set_name (str | None): Name of the example set to use.
                If None, uses the current training_set, else testing_set.
            example_index (int | None): If given, runs that specific example.
                If None, runs the next example via the iterator.
            test (bool): Whether to run in testing mode (passed to standard_net_train_example).
        """
        # 1) Select the example set
        if example_set_name:
            if example_set_name not in self.loaded_example_sets:
                raise ValueError(f'Example set "{example_set_name}" is not loaded.')
            ex_set = self.loaded_example_sets[example_set_name]
        else:
            ex_set = self.training_set or self.testing_set
            if ex_set is None:
                raise ValueError("No active training or testing set is selected.")

        # Select the example
        if example_index is None:
            # get next example using iterator
            example = ex_set.iterate_example()
        else:
            # get specific example directly
            if not (0 <= example_index < ex_set.num_examples):
                raise IndexError(
                    f"example_index {example_index} out of range (0–{ex_set.num_examples - 1})"
                )
            example = ex_set.example[example_index]

        # 3) Run the example through the network
        return self.standard_net_train_example(example, test=test)

    def package_weights(self):
        """
        Packages the weights of the network into a list of lists
        """
        packaged_weights = []

        for j in range(len(self.groups)):
            group_weights = []
            for k in range(len(self.groups[j].incoming_links)):
                group_weights.append([self.groups[j].incoming_links[k].weights])
            packaged_weights.append(group_weights)

        return packaged_weights

    def combine_worker_result(self, workers_results):
        """
        Combines the results of each worker after parallelizing
        
        Args:
            workers_results (List): a list where each element corresponds to a worker's results.
        """
        self.batch_errors = []
        for result in workers_results:
            self.batch_errors += result[1]

        for i in range(len(workers_results)):
            for j in range(len(self.groups)):
                for k in range(len(self.groups[j].incoming_links)):
                    self.groups[j].incoming_links[k].weight_derivs += workers_results[i][2][j][k]

        self.batch_unit_costs = []
        for result in workers_results:
            self.batch_unit_costs += result[3]

    def standard_net_train_batch(self, batch_size, test=False, stop_event=None):
        """
        Iterates over a single batch

        Args:
            batch_size (int): the number of examples in a single batch (if set to 0, the entire example set is run)
            test (boolean): whether the batch is for testing or training
        """
        if test:
            if self.testing_set:
                self.example_sets = [self.testing_set]
            elif self.testing_sets:
                self.example_sets = self.testing_sets
            # In case someone forgets testing flag when loading examples and testing from gui
            else:
                raise ValueError(f"Make sure to load testing example set.")
        else:
            if self.training_set:
                self.example_sets = [self.training_set]
            elif self.training_sets:
                self.example_sets = self.training_sets
            else:
                raise ValueError(f"Make sure to load training example set.")
        res = []
        # TODO: Choose which set of examples to use for training / testing
        for example_set_index, example_set in enumerate(self.example_sets):
            if batch_size == 0:
                batch_size = example_set.num_examples

            if example_set.pre_epoch_proc_name is not None:
                example_set.pre_epoch_proc()

            if self.parallel_mode:
                workers_results = []

                # initialize parallel workers when it is at the first run OR num_worker changed
                if self.parallel_workers == None or self.num_worker != len(self.parallel_workers):
                    self.parallel_workers = [create_parallel_network(
                                             network_class=self.__class__,
                                             name=f"{self.name}_worker{_}", 
                                             time_intervals=self.time_intervals, 
                                             ticks_per_interval=self.ticks_per_interval, 
                                             learning_rate=self.learning_rate, 
                                             add_bias=self.add_bias,
                                             baseType=self.baseType,
                                             ) for _ in range(self.num_worker)]
                    
                    for i in range(self.num_worker):
                        self.parallel_workers[i].replay_initialization_for_workers.remote(self.initialization_actions)

                # send weights to workers
                packaged_weights = self.package_weights()
                for i in range(self.num_worker):
                    self.parallel_workers[i].sync_weights.remote(packaged_weights)

                # calculate how this batch being split to different workers
                examples_per_worker = math.ceil(batch_size / self.num_worker)

                # looping through each worker with their examples
                batch_start = example_set.example_iterator.curr.value
                for i in range(0, batch_size, examples_per_worker):
                    start_index = example_set.example_iterator.curr.value
                    # calculate the steps over examples for each worker
                    steps = min(examples_per_worker, batch_size - i)
                    workers_results.append(
                            self.parallel_workers[int(i/examples_per_worker)].worker_function.remote(
                                example_set_index, start_index, steps, test)
                            )

                    for _ in range(steps):
                        example = example_set.iterate_example()

                    self.max_example_time = example_set.max_time # I don't know what is this line for

                # retrieve information from parallel workers and integrate them into the main network (mostly the weight derivs)
                workers_results = ray.get(workers_results)
                self.combine_worker_result(workers_results)

            for i in range(0, batch_size):
                if not self.parallel_mode:
                    if batch_size < example_set.num_examples:
                        if i == batch_size - 1:
                            example = example_set.iterate_example()
                            # Do not need to reset example here, ExampleIterator
                            # resets automatically after all examples have been seen
                            # example_set.example_iterator.reset_example_list()
                        else:
                            example = example_set.iterate_example()
                    else:
                        example = example_set.iterate_example()
                    self.max_example_time = example_set.max_time
                    result, training_errors, unit_costs = self.standard_net_train_example(example, test)
                    if self.network_type in ['continuous', 'srbptt']:
                        self.net_train_example_back(example)
                    if self.visualized:
                        self.dispatch("example", stop_event)
                    self.update_graphs(update_no=self.examples_token_trained+i, updates_before=self.examples_token_trained, s="example")
                    if test:
                        example_index = example_set.example.index(example)
                        print(
                            "Example " + str(example_index) + " Test Error: ",
                            example.example_test_error
                        )

                else:
                    # retrieve result and errors from each worker
                    result = workers_results[int(i/examples_per_worker)][0]
                    training_errors = workers_results[int(i/examples_per_worker)][1]
                # else:
                res += result


                if test and (self.test_error_criterion or self.test_group_criterion_reached):
                    return
                self.last_example_trained = example
                # if (network_params.PAR_N_reset_on_example):
                #     self.reset_matrices()


            self.examples_token_trained += batch_size
            if example_set.post_epoch_proc_name is not None:
                example_set.post_epoch_proc()

        if self.network_type == 'continuous':
            for group in self.groups:
                scale = 1 / self.ticks_per_interval
                self.gain *= scale
                for link in group.incoming_links:
                    link.weight_derivs *= scale
        return res


    def save_stats(self, file_path="training_stats.csv"):
        """
        Saves the training statistics to a file
        """
        self.stats_plotter.save_stats(file_path)

    def stop_live_plotting(self):
        """
        Stops the live plotting
        """
        self.stop_event.set()

    def plotting_thread(self, q):
        """
        The plotting thread for the network
        
        Args:
            q (List): the list of data to plot
        """
        while not self.stop_event.is_set():
            if self.stats_plotter.live_update_closed:
                break
            if not len(q):
                continue
            else:
                self.first_update = self.num_update == 1
                # self.stats_plotter.live_graphs(first_update, self.graph_quantity, self.graph_title)
                self.stats_plotter.live_graphs("plot")
            # print("still plotting")

    def dispatch(self, s, stop_event):
        """
        Dispatches the network to update the display
        
        Args:
            s (str): the string to dispatch
        """
        # seperate handller for graph update: 
        try:
            self.simulator.update_display_caller(self.last_example_trained, s)
        except Exception as e:
            logging.info(repr(e))

    def execute_input(self):
        """
        Executes user input. returns true if user wants to stop training.
        """
        stop_training_keywords = ["stop", "quit", "finish"]

        user_input = input("paused. type python code to execute here\n")
        if user_input.strip().lower() in stop_training_keywords:
            self.user_interrupt = True
        else:
            try:
                exec(user_input)
            except NameError:
                print("error\n")
                self.execute_input()

    def set_graph_quantity(self, graph_quantity):
        """
        Sets the quantity of graphs to display
        
        Args:
            graph_quantity (int)
        """
        self.graph_quantity = graph_quantity

    def set_graph_title(self, graph_title):
        """
        Sets the title of the graph
        
        Args:
            graph_title (str)
        """
        self.graph_title = graph_title

    def standard_net_train(self, epochs, batch_size, report_interval, stop_event=None):
        """
        Trains the network for the desired number of epochs
        Training consists of making a prediction, calculating the accuracy of the prediction
        and then updating weights to minimize the error until a global minimum is reached
        and the network can make correct predictions

        Args:
            epochs (int): the amount of time to train one (one epoch = one weight update) 
                        so essentially this is the number of weight updates to make
            batch_size (int): the number of examples to run before a weight update 
                        (if set to 0, the entire example set is run)
            report_interval (int): the number of updates between each report
            stop_event (Event): the event to stop the training
        """
        
        training_start_time = time()
        self.stats_plotter.set_training_start_time(training_start_time)
        self.stats_plotter.reset_stats()
        q = []

        if self.plot_thread_created:
            t1 = Thread(target=self.plotting_thread, daemon=True, args=[q])
            t1.start()
            self.plot_thread_created = True

        i = 0
        batches_at_criterion = 0
        reached_batch_error_criterion = False
        self.user_interrupt = False
        self.res = ""
        update_no_list = list()

        if epochs is None:
            epochs = self.num_updates

        while i < epochs and not self.criterion_reached and not self.user_interrupt:
            if self.use_keyboard and keyboard.is_pressed('p'):
                self.stats_plotter.count_run(i)

                self.execute_input()
                if self.user_interrupt:
                    for graph in self.graphs:
                        graph.update_trace()
                    break
            # Reset weight derivs moved to beginning of iteration to keep weight derivs values for GUI
            self.optimizer.reset_weight_derivs()
            self.reset_matrices()
            # reset weight derivs for parallel workers
            if self.parallel_mode == True and self.parallel_workers != None:
                for num in range(len(self.parallel_workers)):
                    self.parallel_workers[num].reset_weight_derivs.remote()
                    self.parallel_workers[num].reset_matrices.remote()

            self.standard_net_train_batch(batch_size)

            # collecting errors and weights for each epoch for debugging purpose
            if self.debug == True:
                self.debug_errors.append(sum(self.batch_errors))
                self.debug_weights.append(self.groups[-1].incoming_links[0].weights[0][0])

            update_no = i + 1
            if self.visualized:
                self.dispatch("completion of a batch", stop_event)
            self.update_graphs(update_no, updates_before=self.num_update, s="completion of a batch")

            if update_no % report_interval == 0 or \
                    update_no == 1 or \
                    update_no == epochs:
                self.update_weights(report_request=True)
                if self.visualized:
                    self.dispatch("weight update", stop_event)
                self.update_graphs(update_no, updates_before=self.num_update, s="weight update")

                # print("In report progress")
                self.stats_plotter.report_progress(update_no=update_no, total_updates=epochs,
                                                   first_update=update_no == 1, plot_queue=q)
                update_no_list.append(update_no)

                if self.visualized:
                    self.dispatch("progress report", stop_event)
                self.update_graphs(update_no, updates_before=self.num_update, s="progress report")

            else:
                self.update_weights()
                if self.visualized:
                    self.dispatch("weight update", stop_event)
                self.update_graphs(update_no, updates_before=self.num_update, s="weight update")

            if self.batch_error(self.batch_errors) <= self.batch_error_threshold or self.group_criterion_reached:
                batches_at_criterion += 1
            else:
                batches_at_criterion = 0

            if batches_at_criterion >= self.min_criterion_batches > 0:
                reached_batch_error_criterion = True
                self.criterion_reached = True

            self.errors = None
            self.batch_errors = None
            self.error_derivs = None
            self.batch_unit_costs = None
            self.unit_cost = None
            self.unit_cost_derivs = None
            i += 1

            if self.use_keyboard and keyboard.is_pressed('p'):
                self.stats_plotter.count_run(i)

                self.execute_input()
                if self.user_interrupt:
                    for graph in self.graphs:
                        graph.update_trace()
                    break
        if self.user_interrupt:
            self.simulator.training_stop_complete()
        if self.visualized:
            self.dispatch("training and testing", stop_event)
        self.update_graphs(update_no, updates_before=self.num_update, s="training and testing")
        self.stats_plotter.count_run(i)

        self.num_update += update_no

        if reached_batch_error_criterion:
            print("Network reached overall error criterion of {}", self.batch_error_threshold)
        if self.group_criterion_reached:
            print("Network reached group output criterion")
        print("Training stopped at epoch:", i)
        print("Total time elapsed:")
        total_time = time() - self.stats_plotter.report_stats["training_start_time"]
        if total_time < 1:
            time_with_units = "{:.5f} s".format(total_time)
            print("{:^20}".format(time_with_units))
        else:
            print("{:^20}".format(format_timespan(total_time)) + "\n")
        return total_time

    def update_graphs(self, update_no, updates_before, s):
        update_events = {
            0: "example",
            1: "weight update",
            2: "completion of a batch",
            3: "progress report",
            4: "training and testing"
        }
        for graph in self.graphs:
            # headless graph deletion keeps None holes for stable IDs
            if graph is None:
                continue
            if update_events[graph.update_after] != s:
                continue
            try:
                if s == update_events[3]:
                    graph.set_clock(updates_before + update_no)
                else:
                    graph.update_clock(1)
                self.send_data_to_graph_viewer(graph)
            except Exception as e:
                warning = f"Graph deprecated: Graph {graph.window_name}\n" + \
                        f"Epoch: {update_no + updates_before}\n" + \
                        "Error message: " + str(e)
                warnings.warn(warning)
                if self.visualized:
                    self.simulator.graph_viewer_warn(warning)

    def send_data_to_graph_viewer(self, graph):
        """
        Sends data to GUI's graph plotter (GraphViewer) OR headless graphs.

        GUI graphs:
        - have one sympy expression graph.plot_variable
        - append into graph.plot_data[-1]

        Headless graphs:
        - have graph.traces: list of trace objects with fields: active: bool, expr: sympy expr, y: list[float]
        - graph.update_x_data(...) appends x to each active trace
        """
        # Always advance x for this sample
        graph.update_x_data(graph.graph_clock)

        # -----------------------------
        # Prepare data cache once
        # -----------------------------
        try:
            data_cache = {
                key: value[-1] if isinstance(value, list) else value
                for key, value in self.stats_plotter.progress_stats.items()
            }
        except Exception:
            data_cache = {
                key: value[-1]
                for key, value in self.stats_plotter.last_progress_stats.items()
            }

        # Add any special variables (idx_* mapped paths etc.)
        for speicial_var, name in getattr(graph, "special_variables", {}).items():
            try:
                val = eval("self." + name)
            except Exception:
                raise ValueError(f"Invalid variable: {name} cannot be processed.")
            if val is None:
                warnings.warn(f"Variable '{name}' is None, replaced with value 0")
                val = 0
            data_cache[str(speicial_var)] = val

        def dynamic_fetch(symbol):
            """
            Fetch the value of a symbol dynamically from stats_plotter or input_net.
            """
            symbol_str = str(symbol)
            if symbol_str in data_cache:
                val = data_cache[symbol_str]
            elif hasattr(self, symbol_str):
                val = getattr(self, symbol_str)
            else:
                raise ValueError(f"Symbol '{symbol_str}' not found in stats_plotter or input_net.")
            try:
                return float(val)
            except Exception:
                raise ValueError(f"Symbol '{symbol_str}'s data type {type(val)} is not supported.")

        # ============================================================
        # HEADLESS MULTI-TRACE PATH (does NOT affect GUI semantics)
        # ============================================================
        if hasattr(graph, "traces"):
            for tr in graph.traces:
                if tr is None:
                    continue
                if not getattr(tr, "active", True):
                    continue
                expr = getattr(tr, "expr", None)
                if expr is None:
                    continue

                try:
                    expr = sp.sympify(expr)
                    symbols = expr.free_symbols
                    subs = {s: dynamic_fetch(s) for s in symbols}

                    result = expr.subs(subs)
                    try:
                        float_result = float(result)
                    except Exception:
                        # if sympy returned something like zoo, nan, etc.
                        raise ValueError(SYM_ERROR_MESSAGES.get(result, result))

                    tr.y.append(float_result)

                except Exception as e:
                    warnings.warn(
                        f"HeadlessGraph trace update failed:\n"
                        f"Graph: {getattr(graph, 'window_name', graph)}\n"
                        f"Trace expr: {getattr(tr, 'expr', None)}\n"
                        f"Error: {e}"
                    )
                    tr.active = False

            # Let the graph decide its own bounds logic
            graph.update_xy_limits()

        # ============================================================
        # GUI SINGLE-EXPR PATH (your existing behavior)
        # ============================================================
        else:
            if graph.plot_variable:
                # Extract symbols from the plot variable
                symbols = sp.sympify(graph.plot_variable).free_symbols

                result = graph.plot_variable.subs({symbol: dynamic_fetch(symbol) for symbol in symbols})
                try:
                    float_result = float(result)
                except Exception:
                    raise ValueError(SYM_ERROR_MESSAGES[result])

                graph.plot_data[-1].append(float_result)

            else:
                # NOTE: this is your original behavior; it looks a bit odd because it appends
                # a scalar into plot_data, but leaving it unchanged to avoid breaking GUI.
                graph.plot_data.append(self.stats_plotter.progress_stats["error"][-1])

            # Change graph limits only if sent data surpasses current limits
            if isinstance(graph.plot_data, list) and graph.plot_data and isinstance(graph.plot_data[-1], list) and graph.plot_data[-1]:
                if graph.plot_data[-1][-1] > graph.y_max:
                    graph.y_max = graph.plot_data[-1][-1]
                if graph.plot_data[-1][-1] < graph.y_min:
                    graph.y_min = graph.plot_data[-1][-1]

            graph.update_xy_limits()

        # -----------------------------
        # Common x-axis window logic
        # -----------------------------
        if graph.graph_clock > graph.x_max:
            current_range = graph.x_max - graph.x_min
            jump = graph.graph_clock - graph.x_max
            margin = max(0.1 * current_range, 0.1 * jump)

            graph.x_max = graph.graph_clock + margin
            graph.update_xy_limits()

        if graph.x_min != graph.min_x_data:
            graph.x_min = graph.min_x_data
            graph.update_xy_limits()

    def reset_output(self):
        """
        Resets the output of the network
        """
        self.res = ""

    def reset_network(self):
        """
        Reset network
        """
        self.optimizer.reset_weights()
        self.optimizer.reset_weight_derivs()
        self.reset_derivs()
        self.reset_output()
        self.num_update = 0
        # Restart plotting from new data
        for graph in self.graphs:
            graph.restart()
            graph.update_trace()
        # Reset gain
        self.gain = network_params.PAR_N_gain

    def save_output(self, filename):
        """
        Saves the output of the network to a file
        
        Args:
            filename (str): the name of the file to save the output to
        """
        summary_file = codecs.open(filename, "w", 'utf-8')
        summary_file.write(self.res)
        summary_file.close()

    def train(self, epochs=network_params.PAR_N_numUpdates, batch_size=network_params.PAR_N_batchSize,
              report_interval=network_params.PAR_N_reportInterval, stop_event=None, parallel_mode=False, num_worker=None):
        """
        Starts training the network for the desired number of epochs with the given batch size.
        Runs sanity checks on the model / optimizer parameters, then trains and prepares progress reporting.

        Args:
            epochs (int): the number of weight updates (epochs) to perform during training
            batch_size (int): the number of samples to include in each batch during training
            report_interval (int): the number of epochs between each progress report
            stop_event (Event): the event to stop the training
            parallel_mode (boolean): whether to train in parallel mode

        Returns:
            total_time (float): the total time taken to train the network
        """
        # Sanity checks
        assert epochs > 0
        assert batch_size > -1
        assert self.training_set
        assert self.learning_rate > 0.0
        assert 0.0 <= self.optimizer.momentum < 1.0
        assert 0.0 <= self.optimizer.weight_decay < 1.0
        assert report_interval > 0

        self.check_params()
        self.parallel_mode = parallel_mode
        if self.parallel_mode:
            if num_worker is not None:
                self.num_worker = num_worker
            print(f"\nParallel training with {self.num_worker} workers")

            warnings.warn(
                """\033[1m
Directly setting properties of a parallel network is not supported, e.g.:
    xor.name = "good"

Instead, use set_properties(), e.g.:
    xor.set_properties(name="good")
            \033[0m""",
                UserWarning
            )
            

        self.report_interval = report_interval

        print("Performing {} updates with batch size {} using {}".format(epochs, batch_size, self.update_method + "\n"))
        self.stats_plotter._prepare_for_report()
        cpu_monitor = CPU().start() if self.parallel_mode else None


        total_time = self.standard_net_train(epochs, batch_size, report_interval, stop_event=stop_event)

        if cpu_monitor is not None:
            cpu_monitor.report()
        if self.net_res_save_path != None:
            self.save_output(self.net_res_save_path)
        return total_time

    def test(self, num_examples, reset_error=False):
        """
        Test the network for the desired number of examples.

        Also performs validation checks on model parameters and prepares internal
        structures for reporting progress.

        Args:
            num_examples (int): Number of examples to test. If 0, test all
                examples in the testing set.
            reset_error (bool): If True, reset stored error statistics.

        Returns:
            float: Final testing error.
        """
        assert self.testing_set
        assert self.learning_rate > 0.0
        assert 0.0 <= self.optimizer.momentum < 1.0
        assert 0.0 <= self.optimizer.weight_decay < 1.0

        # reset errors and examples
        if reset_error == True: 
            self.batch_test_errors = None
            for example_set in self.example_sets:
                example_set.example_iterator.reset_example_list()
                for i in range(example_set.num_examples):
                    example_set.example[i].example_test_error = 0

        print("Loaded in the testing set")
        self.standard_net_train_batch(batch_size=num_examples, test=True)

        if self.test_error_criterion:
            print("Test Error Criterion Reached. Testing Stopped Early.")

        if self.test_group_criterion_reached:
            print("Test Group Criterion Reached. Testing Stopped Early.")

        print("Testing Completed.")
        print("Final Testing Error: ")
        if self.batch_test_errors is not None:
            testing_error = sum(self.batch_test_errors)
            self.stats_plotter._print_report_field(testing_error, last_field=True)

        return testing_error

    def do_test(self, batch_size):
        """
        Ask user if they want to test the network
        
        See Also:
            def train: For the description of 'batch_size'.
        """
        val = input("Do you want to test the network(Y/N): ")
        if (val.strip()).lower() == "y":
            self.test(batch_size)
        elif (val.strip()).lower() == "n":
            print("Testing Aborted")
        else:
            print("Please reply by typing Y or N")
            self.do_test(batch_size)

    def remaining_time(self, update_no, total_updates):
        """
        Calculates how much time remains for training to finish
        
        Args:
            update_no (int): the current update number
            total_updates (int): the total number of updates to perform

        Returns:
            time_left (float): the time left for training to finish
        """
        now = time()
        if update_no < 4:
            # Just take the average time spend on each update
            self.stats_plotter.report_stats["time_per_update"] = (now - self.stats_plotter.report_stats["training_start_time"]) / (update_no + 1)
        else:
            # Take the average update time since last report but also take previous time_per_update into account
            try:
                recent_average_update_time = \
                    (now - self.stats_plotter.report_stats["last_report_time"]) / (
                                update_no - self.stats_plotter.report_stats["last_report_update_no"])
            except ZeroDivisionError:
                recent_average_update_time = 0
            self.stats_plotter.report_stats["time_per_update"] = \
                0.5 * (self.stats_plotter.report_stats["time_per_update"] + recent_average_update_time)
        # Update report records to be used for the next report
        self.stats_plotter.report_stats["last_report_update_no"] = update_no
        self.stats_plotter.report_stats["last_report_time"] = now
        time_left = (total_updates - update_no) * self.stats_plotter.report_stats["time_per_update"]
        return time_left

    def batch_error(self, batch_error):
        """
        Computes the overall network error over the last batch of training.
        It is not accumulated across all batches since the last report.

        Args:
            batch_error: a list of numpy arrays, so we have to sum over it twice
            
        Returns:
            batch_err (float): the overall network error
        """

        if batch_error is None:
            return 0.0
        batch_err = sum(batch_error)
        return batch_err
    
    def sum_batch_unit_cost(self, batch_unit_cost):
        """
        Sums up the batch unit cost and updates the progress stats.

        Args:
            batch_unit_cost (list): A list of unit costs for each batch.
        
        Returns:
            batch_cost (float): The sum of the unit costs.
        """
        if batch_unit_cost is None:
            return 0.0
        batch_cost = sum(batch_unit_cost)
        return batch_cost

    def group_error(self):
        """
        Computes the overall network error over the last batch of training.
        It is not accumulated across all batches since the last report.
        
        Returns:
            batch_err (float): the overall network error
        """

        return sum(self.batch_errors)

    def weight_cost(self):
        """
        Returns the sum of the squared weight values i.e. weight cost
        This is calculated while weights were updated

        Returns:
            weight_cost (float): the sum of the squared weight values
        """
        return self.stats_plotter.report_stats["squared_weights"] / 2

    def gradient_linearity(self):
        """
        Computes and returns the gradient linearity using report statistics

        Returns:
            grad_lin (float): the gradient linearity
        """
        grad_lin = - self.stats_plotter.report_stats["lwd_times_derivs"]
        last_delta_len = self.stats_plotter.report_stats["squared_lwd"]
        deriv_len = self.stats_plotter.report_stats["squared_derivs"]
        if last_delta_len * deriv_len == 0:
            return None  # because division by zero is not possible
        else:
            return grad_lin / af.sqrt(last_delta_len * deriv_len)

    def compute_cost(self, output_groups, frequency, tick):
        """
        Computes the cost of the networks prediction

        Args:
            output_groups (Group): these are all of the groups in the network of type output
            frequency (float): frequency of the example for this cost calculation
            tick (int): current network tick
        
        Returns:
            error_groups (List)
            error_derivs (List)
        """
        error_groups = []
        error_derivs = []

        for i, output in enumerate(output_groups):
            target = output.target
            cost = self.cost_functions[i]

            error, adj_targets = cost.forward(
                output.output_matrix,
                target,
                frequency,
            )
            error_groups.append(error)
            error_derivs.append(
                cost.backward(
                    output.output_matrix,
                    adj_targets,
                    frequency,
                )
            )

            if self.parallel_mode:
                output.target_history = output.target_history.copy()

            output.target_history[tick] = target

        return error_groups, error_derivs

    def compute_unit_output_cost(self, output_groups):
        """
        Computes the unit output cost of the network
        
        Args:
            output_groups (List): the output groups of the network
            
        Returns:
            unit_cost_groups (List): the unit cost groups
            unit_cost_derivs (List): the unit cost derivates
        """
        unit_cost_groups = []
        unit_cost_derivs = []
        for i, output_group in enumerate(output_groups):
            unit_cost = self.unit_cost_functions[i]
            if unit_cost is None:
                unit_cost_groups.append(0.0)
                unit_cost_derivs.append(af.zeros_like(output_group.output_matrix))
                continue
            cost = unit_cost.forward(output_group.output_matrix)
            cost_derivs = unit_cost.backward(output_group.output_matrix)
            unit_cost_groups.append(cost)
            unit_cost_derivs.append(cost_derivs)
        return (unit_cost_groups, unit_cost_derivs)
    
    def compute_back(self):
        """
        Propogate the error derivative into the output groups and calculate the rest of the network derivates
        
        Returns:
            out (List): the output of the network
        """

        for i in range(len(self.output_groups)):
            self.output_groups[i].output_derivs = self.error_derivs[i] + self.unit_cost_derivs[i]
        out = self.backward()
        return out

    def reset_outputs(self, group):
        """
        Resets the cache
        
        Args:
            group (Group): the group to reset the cache 
        """
        init_output = self.initOutput # need to implement init_output choose value later
        # TODO change this to work for nan external inputs
        if group.group_type in ('elman', 'input'):
            if group.external_input is not None:
                group.output_matrix[:] = group.external_input
            else:
                af.fill(group.output_matrix, init_output)
            group.output_matrix_cache = af.copy(group.output_matrix)
        elif group.group_type == 'bias':
            pass
        else:
            # Make a copy of output matrix to fix "assignment destination is read-only"
            if self.parallel_mode:
                group.output_matrix = af.copy(group.output_matrix)
            af.fill(group.output_matrix, init_output)
            group.output_matrix_cache = af.copy(group.output_matrix)

    def reset_forwardintegrators(self, group):
        """
        Resets the forward integrators
        
        Args:
            group (Group): the group to reset the forward integrators
        """
        for transform in group.output_transforms:
            if transform.name == 'Out_Integr':
                if self.parallel_mode:
                    transform.unitData = transform.unitData.copy()
                af.fill(transform.unitData, group.initOutput)
        for transform in group.input_transforms:
            if transform.name == 'In_Integr':
                af.fill(transform.unitData, self.initInput)

    def store_links_to_dict(self, links, dictionary, weight_only):
        """
        Stores the links in a dictionary
        
        Args:
            links (List): the links to store
            dictionary (Dict): the dictionary to store the links in
            weight_only (boolean): whether to store only the weights
        """
        for link in links:
            link_dict = {'weights': link.to_json()['weights'], 'outgoing_group': link.to_json()['outgoing_group'],
                         'incoming_group': link.to_json()['incoming_group'],
                         'proj_type': link.to_json()['proj_type']} if weight_only else link.to_json()
            connection_name = "{}>{}".format(link_dict["outgoing_group"], link_dict["incoming_group"])
            if connection_name not in dictionary:
                dictionary[connection_name] = [link_dict]
            else:
                if link_dict not in dictionary[connection_name]:
                    dictionary[connection_name].append(link_dict)

    def to_json(self):
        """
        Converts the network to a json object
        
        Returns:
            result (Dict): the network as a json object
        """
        result = {}
        for name, data in self.__dict__.items():
            if isinstance(data, int) or isinstance(data, float) or isinstance(data, str) or isinstance(data, bool):
                result[name] = data
        return result

    def from_json(self, data):
        """
        Converts the network from a json object
        
        Args:
            data (Dict): the json object to convert from
        """
        for key in data:
            setattr(self, key, data[key])

    def store_links(self, fn, weight_only=False, format='pickle'):
        """
        Stores the links of the network to a file
        
        Args:
            fn (str): the name of the file to load
            weight_only (boolean): whether to store only the weights
            format (str): the format to store the links in
        """
        connection2links = {}
        for group in self.groups:
            group_name = group.name
            self.store_links_to_dict(group.outgoing_links, connection2links, weight_only)
            self.store_links_to_dict(group.incoming_links, connection2links, weight_only)
            # incoming_links = [link.to_json() for link in group.incoming_links]
        if format == 'pickle':
            with open(fn, 'wb') as f:
                pickle.dump(connection2links, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "json":
            with open(fn, 'w') as f:
                json.dump(connection2links, f, indent=4, default=LinkFactory.store_data_type_converter)

    def load_links(self, fn):
        """
        Loads the links of the network from a file
        
        See Also:
            def store_links: For the description of 'fn'.
        """
        try:
            if fn.endswith('pickle'):
                with open(fn, 'rb') as f:
                    connection2links = pickle.load(f)
            elif fn.endswith('json'):
                with open(fn, 'r') as f:
                    connection2links = json.load(f)
        except:
            return
        visited = {}  # group_name: idx
        for connection_name in connection2links:
            outgoing_group_name, incoming_group_name = connection_name.split(">")
            outgoing_group, incoming_group = self._group_pair_from_names(outgoing_group_name, incoming_group_name)
            if not outgoing_group or not incoming_group:
                raise Error("No link in current architecture match stored link")
            for store_obj in connection2links[connection_name]:
                link = LinkFactory.load_into_new_link(outgoing_group, incoming_group, store_obj)
                group_idx = self._group_idx_eq_link(outgoing_group.outgoing_links, link, outgoing_group_name, visited)
                LinkFactory.load_into_given_link(outgoing_group.outgoing_links[group_idx], outgoing_group,
                                                 incoming_group, store_obj)

    def store_groups(self, fn, weight_only=False, format='pickle'):
        """
        Stores the groups of the network to a file
        
        See Also:
            def store_links: For the description of 'fn' and 'format'.
        """
        group_parameters = {}
        for group in self.groups:
            group_parameters[group.name] = group.to_json()
        print(group_parameters)

        if format == 'pickle':
            with open(fn, 'wb') as f:
                pickle.dump(group_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "json":
            with open(fn, 'w') as f:
                json.dump(group_parameters, f, indent=4, default=LinkFactory.store_data_type_converter)

    def load_groups(self, fn):
        """
        Loads the groups of the network from a file
        
        See Also:
            def store_links: For the description of 'fn'.
        """
        try:
            if fn.endswith('pickle'):
                with open(fn, 'rb') as f:
                    group_parameters = pickle.load(f)
            elif fn.endswith('json'):
                with open(fn, 'r') as f:
                    group_parameters = json.load(f)
        except:
            return
        for group in self.groups:
            group.from_json(group_parameters[group.name])

    def store_network(self, fn, weight_only=False, format='pickle'):
        """
        Stores the network to a file

        See Also:
            def store_links: For the description of 'fn' and 'format'.
        """
        network_meta = self.to_json()
        # network_meta['num_update'] = self.num_update

        if format == 'pickle':
            with open(fn, 'wb') as f:
                pickle.dump(network_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "json":
            with open(fn, 'w') as f:
                json.dump(network_meta, f, indent=4, default=LinkFactory.store_data_type_converter)

    def load_network(self, fn):
        """
        Loads the network from a file
        
        See Also:
            def store_links: For the description of 'fn'.
        """
        try:
            if fn.endswith('pickle'):
                with open(fn, 'rb') as f:
                    network_meta = pickle.load(f)
            elif fn.endswith('json'):
                with open(fn, 'r') as f:
                    network_meta = json.load(f)
        except:
            return
        self.from_json(network_meta)
        # self.num_update = network_meta['num_update']

    def load_clen_weight(self, fn):
        """
        Loads the weights of the network from a file
        
        See Also:
            def store_links: For the description of 'fn'.
        """
        with open(fn, 'r') as f:
            # read the header lines
            for i in range(4):
                f.readline()

            for group in self.groups[1:]:
                for out_unit_idx in range(group.num_units):
                    i = 0
                    for link in group.incoming_links:
                        i += 1

                        link_alpha = []
                        if isinstance(link, LinkOneToOne):
                            # if link type is elman, skip over reading weights,
                            # ie. Assign weight = 1, delta = 0, alpha = 1

                            # if link type is not elman
                            if link.incoming_group.group_type != "elman":
                                try:
                                    wt = [eval(num) for num in f.readline().strip().split()]
                                    if len(wt) == 1:
                                        wt = [wt[0], 0, 1]
                                    elif len(wt) == 2:
                                        wt = [wt[0], wt[1], 1]
                                    link_alpha.append(wt[2])
                                    if len(link.weights.shape) == 1:
                                        link.weights[out_unit_idx] = wt[0]
                                        link.last_weight_delta[out_unit_idx] = wt[1]
                                    else:
                                        link.weights[in_unit_idx, out_unit_idx] = wt[0]
                                        link.last_weight_delta[in_unit_idx, out_unit_idx] = wt[1]
                                except:
                                    print("incorrect weight file")

                        else:
                            for in_unit_idx in range(link.weights.shape[0]):
                                try:
                                    wt = [eval(num) for num in f.readline().strip().split()]

                                    if len(wt) == 1:
                                        wt = [wt[0], 0, 1]
                                    elif len(wt) == 2:
                                        wt = [wt[0], wt[1], 1]
                                    link_alpha.append(wt[2])
                                    if len(link.weights.shape) == 1:
                                        link.weights[out_unit_idx] = wt[0]
                                        link.last_weight_delta[out_unit_idx] = wt[1]
                                    else:
                                        link.weights[in_unit_idx, out_unit_idx] = wt[0]
                                        link.last_weight_delta[in_unit_idx, out_unit_idx] = wt[1]
                                except:
                                    print("incorrect weight file")

                                # print("wt: {}".format(wt))
                            link_alpha = sum(link_alpha) / len(link_alpha)
                            link.alpha = link_alpha

                            # print("{} ---> from {} {} to {} {}".format(wt, link.outgoing_group.name, in_unit_idx, group.name, out_unit_idx))

    def load_clen_binary_weigth(self, fn):
        """
        Loads the weights of the network from a file
        
        See Also:
            def store_links: For the description of 'fn'.
        """
        import struct
        with open(fn, 'rb') as f:
            # read magic number and other
            print(struct.unpack('i', f.read(4)))
            print(struct.unpack('i', f.read(4)))
            print(struct.unpack('i', f.read(4)))
            print(struct.unpack('i', f.read(4)))
            # read weights
            print(struct.unpack('f', f.read(4)))
            print(struct.unpack('f', f.read(4)))
            print(struct.unpack('f', f.read(4)))
            print(struct.unpack('f', f.read(4)))

    def store_weight(self, fn, weight_only=False, format='pickle', storage_objects=["weights", "links", "groups", "network"]):
        """
        Store network weights (and optionally other objects) into a file.

        Selected objects are serialized and saved using the specified format.

        Example storage object dictionary::

            {
                'link_alpha': True,
                'link_lesion_rate': False,
                'link_dropout_rate': False,
                'link_perma_lesion_rate': False,
                'unit_alpha': False,
                'unit_lesion_rate': False,
                'unit_dropout_rate': False,
                'unit_perma_lesion_rate': False,
                'network_alpha': True
            }

        A value of ``False`` means the corresponding field is not stored.

        Args:
            fn (str): Output filename.
            weight_only (bool): If True, only store weight matrices.
            format (str): Serialization format (e.g., ``"pickle"``).
            storage_objects (list): Flags indicating which objects to store.

        See Also:
            store_links: For details on parameters ``fn``, ``weight_only``,
            and ``format``.
        """
        if "weights" in storage_objects:
            self.store_links(fn + '_weights.' + format, weight_only=weight_only, format=format)

        # Not implemented in gui:
        if "links" in storage_objects:
            self.store_links(fn + '_links.' + format, weight_only=weight_only, format=format)
        if "groups" in storage_objects:
            self.store_groups(fn + '_groups.' + format, weight_only=weight_only, format=format)
        if "network" in storage_objects:
            self.store_network(fn + '_network.' + format, weight_only=weight_only, format=format)

    def load_weight(self, fn):
        # TODO: error check for same architecture, check for subset relationship
        # TODO: test for overiding dense wtih sparse
        # TODO: add meta values, total number of updates
        """
        Load weights to network links from given filename
        All link in the stored network must be a subset of links in current architecture

        See Also:
            def store_links: For the description of 'fn'.

        Raises:
            IOError: if the file is not in the correct format
        """
        if fn.endswith(".wt"):
            self.load_clen_weight(fn)
        elif fn.endswith(".pickle"):
            self.load_links(fn)
        else:
            raise IOError("Could not read weight file. Please make sure it is in the correct format.")

    def _group_idx_eq_link(self, links, link, group_name, visited):
        """
        Find the index of the group that matches the link
        
        Args:
            links (List): the links to check
            link (Link): the link to match
            group_name (str): the name of the group to match
            visited (Dict): the visited groups
            
        Returns:
            i (int): the index of the group that matches the link
            
        Raises:
            Error: if no link in the current architecture matches the stored link
        """
        if group_name not in visited:
            visited[group_name] = []
        for i, cur_link in enumerate(links):
            if link == cur_link and i not in visited[group_name]:
                visited[group_name].append(i)
                return i
        raise Error("No link in current architecture match stored link")

    def check_params(self):
        """
        Checks the parameters of the network
        """
        if self.gain < 0 or isinstance(self.gain, list):
            print("gain value is not a positive number or network gain is list instead of an int")
            sys.exit("gain value is not a positive number or network gain is list instead of an int")
        elif self.noise_range < 0:
            sys.exit("noise_range has to be a positive number")
        elif self.clamp_strength > 1 or self.clamp_strength < 0:
            sys.exit("clamp_strength should be between 0 and 1")

        if self.gain == 0:
            print("Warning: Network gain value has reached zero.")

        for group in self.groups:
            group.check_params()

class CPU:
    """CPU core idle time for *this process* between start() and stop()."""
    def __init__(self, label=""):
        self.cpu_time0 = None
        self.wall_time0 = None

    def start(self):
        self.cpu_time0 = _time.process_time()
        self.wall_time0 = _time.perf_counter()
        return self

    def stop(self):
        wall_time = _time.perf_counter() - self.wall_time0
        cpu_time  = _time.process_time() - self.cpu_time0
        avg_one_core = (cpu_time / wall_time * 100.0) if wall_time > 0 else 0.0
        return avg_one_core, wall_time, cpu_time

    def report(self, prefix="[Main process idle time]"):
        avg_one_core,  wall_time, cpu_time = self.stop()
        print(f"{prefix}: {100-avg_one_core:.1f}% of 1 core ")
        # return avg_one_core,  wall_time, cpu_time
