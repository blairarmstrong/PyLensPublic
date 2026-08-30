import copy
from typing import List, Union

from ..examples.example import Example
from ..examples.event import Event
from .group import Group
from .pytorch_group import PytorchGroup

from .parameters import NetworkParameters
from .array_factory import Array_factory as af
from .network import Network
from .output_transforms.boltzmann import BoltzmannOutput

network_params = NetworkParameters()

class BoltzmannMachine(Network):

    """
    This class implements a deterministic Boltzmann machine, which is trained through minimizing the difference between negative and positive phases of the network.

    """
    def __init__(self, 
                 name,
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
                 pseudo_example_freq=None):

        self.network_type = 'boltzmann'

        super().__init__(name,
                         baseType=baseType,
                         time_intervals=time_intervals,
                         ticks_per_interval=ticks_per_interval,
                         add_bias=add_bias,
                         learning_rate=learning_rate,
                         batch_error_threshold=batch_error_threshold,
                         group_criterion_threshold=group_criterion_threshold,
                         num_updates=num_updates,
                         min_criterion_batches=min_criterion_batches,
                         update_method=update_method,
                         stats_plotted=stats_plotted,
                         pseudo_example_freq=pseudo_example_freq)


        # Boltzmann Machine parameters
        self.in_grace_period = True
        self.init_gain = network_params.PAR_N_initGain
        self.final_gain = network_params.PAR_N_finalGain
        self.anneal_time = network_params.PAR_N_annealTime
        self.ticks_per_event = []

    @property
    def event_list(self)->list:
        """
        Retrieves a list containing the event at each tick for visualization in the GUI.

        Returns:
            list: A list where each index represents a tick, and the value represents the event.
        """
        event_list = []
        if self.ticks_per_event:
            for i, ticks in enumerate(self.ticks_per_event):
                event_list += [i] * ticks
        # In case network hasn't seen an example and GUI is called
        else:
            event_list = [0]
        return event_list

    def add_group(self,
                  num_units: int,
                  name=None,
                  group_type="hidden",
                  input_transforms: list = None,
                  output_transforms: list = None,
                  error_function=None,
                  lesion_rate=None,
                  dropout_rate=None,
                  num_cols=None,
                  biased=None,
                  unit_cost_function=None):
        """
        Adds a new group of units to the Boltzmann Machine.

        Args:
            num_units (int): The number of units in the new group.
            name (str, optional): The name of the group. If None, a default name is assigned.
            group_type (str, optional): The type of the group ("input", "hidden", "output", "bias").
            input_transforms (list, optional): List of input transformation functions.
            output_transforms (list, optional): List of output transformation functions.
            error_function (str, optional): The error function to be applied if the group is an output group.
            lesion_rate (float, optional): The lesion rate for the group (used for disabling units).
            dropout_rate (float, optional): The dropout rate for the group (used for regularization).
            num_cols (int, optional): The number of columns if the group represents a structured layout.
            biased (bool, optional): Whether the group should be biased (i.e., linked to a bias group).
            unit_cost_function (str, optional): The unit-wise cost function applied to this group.

        Raises:
            ValueError: If an unknown `group_type` is provided.
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

        # All transforms default to boltzmann input/output transforms for output/hidden groups.
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

        # check if name exists
        if self.check_name(name):
            # instantiate a new group object, append it to master list of groups
            new_group = Group(name, num_units, group_type, input_transforms, output_transforms, self.time_intervals,
                              self.ticks_per_interval, self, num_cols=num_cols)

            if (group_type not in ("input", "bias", "elman") and biased is None):
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
                    error_function = "cross_entropy"
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
                self.bias.initOutput = self.initOutputBias
                self.bias.output_matrix = af.array([self.bias.initOutput])

    def boltzmann_update(self, tick: int) -> list:
        """
        Performs one step of deterministic unit updates in the Boltzmann Machine.

        Args:
            tick (int): The current tick of the example.

        Returns:
            list: The updated output matrices for each group.
        """
        group_outputs = []

        # compute input for all groups
        for group in self.groups:
            group.curr_tick = tick - 1
            group.compute_input()

            if group.group_type != "bias": # Bias group does not have input history
                group.input_history[tick - 1] = group.input_matrix

        # compute output for all groups
        for group in self.groups:
            group.compute_output()
            if group.lesion_mask is not None:
                group.output_matrix *= group.lesion_mask
            # reinitialize dropout mask
            group.unit_dropout(group.dropout_rate)
            if group.dropout_mask is not None:
                group.output_matrix *= group.dropout_mask
            group.input_set = False
            group_outputs += [group.output_matrix]
            if group.group_type != "bias": # Bias group does not have output history
                group.output_history[tick - 1] = group.output_matrix
                group.target_history[tick - 1] = group.target

        return group_outputs

    def boltzmann_settled(self, training: bool) -> bool:
        """
        Checks if the Boltzmann Machine has converged, meaning that unit outputs 
        have not changed beyond a given criterion.

        Args:
            training (bool): Indicates whether the network is in training mode.

        Returns:
            bool: True if the network has settled, otherwise False.
        """
        if training:
            criterion = self.group_criterion_threshold
        else:
            criterion = self.test_group_criterion_threshold
        for group in self.groups[:]:
            for output_transform in group.output_transforms:
                if isinstance(output_transform, BoltzmannOutput):
                    last_output = output_transform.unit_data
                    if (abs(group.output_matrix_cache - last_output) > criterion).any():
                        return False
        return True

    def initialize_boltzmann_outputs(self, group: Group) -> None:
        """
        Initializes the outputs of the Boltzmann Machine according to the group type.

        Args:
            group (Group): The group whose outputs need initialization.
        """
        if group.group_type != 'bias':
            for i in range(group.num_units):
                if not af.isnan(group.external_input[i]):
                    group.output_matrix[i] = group.external_input[i]
                elif not af.isnan(group.target[i]):
                    group.output_matrix[i] = group.target[i]
                else:
                    group.output_matrix[i] = self.initOutput
            group.output_matrix_cache = copy.copy(group.output_matrix)

    def reset_boltzmann_outputs(self, group: Group) -> None:
        """
        Resets Boltzmann unit outputs based on the clamp strength, similar to weak clamping.

        Args:
            group (Group): The group whose outputs need resetting.
        """
        clamp_strength = group.clamp_strength if not af.isnan(group.clamp_strength) else self.clamp_strength
        retain_strength = 1.0 - self.clamp_strength
        initOutput = (group.initOutput if group.initOutput is not None else self.initOutput) * clamp_strength

        for i in range(group.num_units):
            if af.isnan(group.external_input[i]):
                group.output_matrix[i] = float(initOutput)  + group.output_matrix[i] * retain_strength
        group.output_matrix_cache = copy.copy(group.output_matrix)

    def standard_net_train_example(self, example, test=False) -> list:
        self.ticks_per_event.clear()
        self.reset_history()
        event_result = []
        if test:
            cur_event_result, errors, unit_costs  = self.boltzmann_net_test_example(example)
            event_result += cur_event_result
        else:
            cur_event_result, errors, unit_costs = self.boltzmann_net_train_example(example)
            event_result += cur_event_result
        return event_result, errors, unit_costs
        
    def boltzmann_net_train_example(self, example: Example) -> list:
        """
        Trains the Boltzmann Machine on a given example using both the positive and negative phases.

        Args:
            example (Example): The training example to use for training.

        Returns:
            list: A list containing the sequence of group outputs during training.
        """
        tick = 1
        ticks_on_phase = 0
        ticks_on_event = 0
        phase_done = False
        time_on_phase = 0.
        min_time = 0.
        max_time = 0.
        grace_time = 0.

        event_result = []
        target_str = ""

        phase = "new_event"
        for event in example.event:
            if phase == "new_event":
                # pre load_input and load_target of event in order to properly reset_output
                self.load_event(event)
                for targ in event.target_group:
                    target_str += ' '.join(map(str, af.astype(targ, int)))
                    target_str += " "
                    
                for group in self.groups[:]:
                    if group.group_type != "bias":
                        self.initialize_boltzmann_outputs(group)

                min_time = event.min_time if event.min_time else example.set.min_time
                max_time = event.max_time if event.max_time else example.set.max_time
                grace_time = event.grace_time if event.grace_time else example.set.grace_time
                neg_phase_time = max_time - grace_time
                ticks_on_phase = 0
                ticks_on_event = 0
                self.in_grace_period = True
                phase = "positive"
            while (tick) <= self.time_intervals * self.ticks_per_interval:
                # Anneal gain
                self.gain_step(ticks_on_phase)

                # Update Boltzmann Machine's units
                event_result += self.boltzmann_update(tick)

                ticks_on_phase += 1 
                ticks_on_event += 1
                time_on_phase = ticks_on_phase / self.ticks_per_interval

                if tick == self.time_intervals * self.ticks_per_interval: 
                    phase_done = True
                elif time_on_phase < min_time:
                    phase_done = False
                elif phase == "positive" and time_on_phase >= grace_time:
                    phase_done = True
                elif phase == "negative" and (time_on_phase >= neg_phase_time):
                    phase_done = True
                elif self.boltzmann_settled(training=True):
                    phase_done = True
                else:
                    phase_done = False

                if phase_done:
                    if phase == "positive":
                        for group in self.groups[:]:
                            # Cache positive-phase outputs for the contrastive Hebbian update.
                            self.cache_outputs_as_derivs(group)
                            if network_params.PAR_N_reset_on_example:
                                self.reset_boltzmann_outputs(group)
                        if ticks_on_phase == self.time_intervals * self.ticks_per_interval:
                            self.ticks_per_event.append(ticks_on_event)
                        ticks_on_phase = 0
                        self.in_grace_period = False
                        phase = "negative"
                        tick += 1
                        event_result += self.store_outputs_and_targets(tick)
                    else:
                        for outg in self.output_groups:
                            # For Boltzmann, this output error is not used for training
                            self.errors, self.error_derivs = self.compute_cost([outg],
                                                                               outg.target,
                                                                               example.frequency,
                                                                               tick)
                            example.example_train_error.append(sum(self.errors))
                            self.unit_cost, self.unit_cost_derivs = self.compute_unit_output_cost([outg])

                            # Accumulate errors over the batch
                            if self.batch_errors is None:
                                self.batch_errors = self.errors
                            else:
                                self.batch_errors = [i + j for i, j in zip(self.batch_errors,
                                                     self.errors)]
                            # Accumulate unit costs over the batch
                            if self.batch_unit_costs is None:
                                self.batch_unit_costs = self.unit_cost
                            else:
                                self.batch_unit_costs = [i + j for i, j in zip(self.batch_unit_costs, 
                                                        self.unit_cost)]

                        # Compute Boltzmann weight derivatives using the contrast between
                        # positive- and negative-phase unit coactivations.
                        for group in self.groups[:]:
                            group.compute_input_back()

                        phase = "new_event"
                        self.ticks_per_event.append(ticks_on_event)
                        tick += 1
                        break
                tick += 1

        self.ticks_on_example = tick
        self.ticks_per_event[-1] += 1 # Add one more tick to last event to match ticks per example
        self.res = str(example.name) + "|output "
        for outg in self.output_groups:
            self.res += ' '.join(map(str, outg.output_matrix)) + " "
        self.res += "\n" + str(example.name) + "|target " + target_str + "\n"
        if example.post_proc_name is not None:
            example.post_proc()

        if self.errors is None:
            self.errors = [0.0]
        if self.unit_cost is None:
            self.unit_cost = [0.0]
            
        return event_result, self.errors, self.unit_cost

    def boltzmann_net_test_example(self, example: Example) -> list:
        """
        Evaluates the Boltzmann Machine on a given example using the **negative phase** only.

        Args:
            example (Example): The testing example.

        Returns:
            list: A list containing the sequence of group outputs during testing.
        """
        tick = 1
        ticks_on_event = 0
        event_done = False
        time_on_event = 0.
        min_time = 0.
        max_time = 0.

        event_result = []
        target_str = ""

        phase = "new_event"
        for event in example.event:
            if phase == "new_event":
                # pre load_input and load_target of event in order to properly reset_output
                self.load_event(event)
                for targ in event.target_group:
                    target_str += ' '.join(map(str, af.astype(targ, int)))
                    target_str += " "
                    
                for group in self.groups[:]:
                    self.initialize_boltzmann_outputs(group)
                    self.cache_outputs_as_derivs(group)
                    if network_params.PAR_N_reset_on_example:
                        self.reset_boltzmann_outputs(group)

                min_time = event.min_time if event.min_time else example.set.min_time
                max_time = event.max_time if event.max_time else example.set.max_time
                grace_time = event.grace_time if event.grace_time else example.set.grace_time
                neg_phase_time = max_time - grace_time
                ticks_on_event = 0
                self.in_grace_period = False
                phase = "negative"
                event_result += self.store_outputs_and_targets(tick)
                tick += 1

            while (tick) <= self.time_intervals * self.ticks_per_interval:
                # Anneal gain
                self.gain_step(ticks_on_event)

                # Update Boltzmann Machine's units
                event_result += self.boltzmann_update(tick)

                ticks_on_event += 1
                time_on_event = ticks_on_event / self.ticks_per_interval

                if tick == self.time_intervals * self.ticks_per_interval: 
                    event_done = True
                elif time_on_event < min_time:
                    event_done = False
                elif phase == "negative" and time_on_event >= neg_phase_time:
                    event_done = True
                elif self.boltzmann_settled(training=False):
                    event_done = True
                else:
                    event_done = False

                if event_done:
                    for outg in self.output_groups:
                        self.test_errors, self.test_error_derivs = self.compute_cost([outg],
                                                                                     outg.target,
                                                                                     example.frequency,
                                                                                     tick)
                        example.example_test_error += (sum(self.test_errors))
                        self.test_unit_cost, self.test_unit_cost_derivs = self.compute_unit_output_cost(self.output_groups)
                        # Accumulate test_test_errors over the batch
                        if self.batch_test_errors is None:
                            self.batch_test_errors = self.test_errors
                        else:
                            self.batch_test_errors = [i + j for i, j in zip(self.batch_test_errors,
                                                      self.test_errors)]
                        # Accumulate unit costs over the batch
                        if self.batch_test_unit_cost is None:
                            self.batch_test_unit_cost = self.test_unit_cost
                        else:
                            self.batch_test_unit_cost = [i + j for i, j in zip(self.batch_test_unit_cost, 
                                                        self.test_unit_cost)]

                    phase = "new_event"
                    self.ticks_per_event.append(ticks_on_event)
                    tick += 1
                    break
                tick += 1

        self.ticks_on_example = tick
        self.ticks_per_event[-1] += 1 # Add one more tick to last event to match ticks per example
        self.res = str(example.name) + "|output "
        for outg in self.output_groups:
            self.res += ' '.join(map(str, outg.output_matrix)) + " "
        self.res += "\n" + str(example.name) + "|target " + target_str + "\n"
        if example.post_proc_name is not None:
            example.post_proc()

        return event_result, sum(self.test_errors), sum(self.test_unit_cost)

    def load_target(self, target_matrix: Union[list, af]) -> None:
        """
        Loads the target values into the output groups.

        Args:
            target_matrix (Union[list, af]): A matrix representing the desired output values.
        """
        target_matrix = af.array(target_matrix)
        for i in range(len(self.output_groups)):
            self.output_groups[i].previous_target(target_matrix[i])
    
    def load_event(self, event: Event) -> None:
        """
        Loads an event’s input and target data into the corresponding input and output groups.

        Args:
            event (Event): The event object containing input and target information.
        """
        self.load_input(event.input_group)
        self.load_target(event.target_group)
        # Add external input to history for unit viewer
        for group in self.groups[:]:
            if group.group_type == "input":
                group.external_input_history.append(group.external_input) 

    def gain_step(self, ticks: int) -> None:
        """
        Adjusts the network gain using exponential decay for annealing.

        Args:
            ticks (int): The number of ticks in the current phase of training.
        """
        self.gain = self.final_gain + (self.init_gain - self.final_gain) \
                    * 0.5 ** (ticks / (self.anneal_time * self.ticks_per_interval))

    def cache_outputs_as_derivs(self, group: Group) -> None:
        """
        Copies the group's current outputs into its output derivative buffer.
        """
        group.output_derivs[:] = group.output_matrix

    def store_outputs_and_targets(self, tick: int) -> list:
        """
        Stores the network's outputs and targets at a specific tick for historical tracking.

        Args:
            tick (int): The current tick of the network.

        Returns:
            list: A list of output matrices for each group.
        """
        group_outputs = []
        for group in self.groups[:]:
            if group.name != 'bias':
                group.output_history[tick - 1] = group.output_matrix
                group_outputs += [group.output_matrix]
                group.input_history[tick - 1] = group.input_matrix
                group.target_history[tick - 1] = group.target

        return group_outputs

    def reset_history(self) -> None:
        """
        Resets the network's historical records, clearing stored outputs, inputs, and targets.
        """
        for group in self.groups[:]:
            if group.name != 'bias':
                group.output_history[:] = af.NaN
                group.input_history[:] = af.NaN
                group.target_history[:] = af.NaN
                if group.group_type == 'input':
                    group.external_input_history.clear()
            else:
                self.output_history = af.array([self.initOutputBias])
