from .network import Network
from .parameters import NetworkParameters
from .parameters import OptimizerParameters
from .array_factory import Array_factory as af


network_params = NetworkParameters()
optimizer_params = OptimizerParameters()


class SRBPTTNetwork(Network):
    """
    This class implements the SRBPTT (Simple Recurrent Backpropagation Through Time) network. It provides functionality for training the recurrent neural network using recurrent backpropagation through time.

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
                 net_res_save_path=None):

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
        self.network_type = 'srbptt'

    def forward(self, tick, target, example):
        """
        Performs a forward pass through the network.

        This method iterates through all groups in the network, computing their
        input and output at the given time step.

        Args:
            tick (int): The current tick of the network.
            target (array-like): The target values for the given input.
            example (object): The example being processed.

        Returns:
            list: A list containing the outputs of each group.
        """

        group_outputs = []
        # iterate through all of the groups and compute the forward pass for each
        # first loop computes input, second compute output

        for group in self.groups:
            group.curr_tick = tick
            if group.group_type != "bias":

                # compute input for a group
                group.compute_input()
                group.input_history[tick] = group.input_matrix

                # compute output for a group
                if group.group_type == 'input':
                    group.output_matrix = group.external_input
                else:
                    group.output_matrix = group.input_matrix
                group.output_matrix = group.input_matrix
                group.compute_output()

                if group.lesion_mask is not None:
                    group.output_matrix *= group.lesion_mask
                # reinitialize dropout mask
                group.unit_dropout(group.dropout_rate)
                if group.dropout_mask is not None:
                    group.output_matrix *= group.dropout_mask
                group.input_set = False

                group.output_history[tick] = group.output_matrix
                group_outputs += [group.output_matrix]

                # Comput cost
                self.errors, self.error_derivs = self.compute_cost(
                    self.output_groups, target, example.frequency, tick)


        return group_outputs

    def net_train_example_back(self, example):
        """
        Performs backpropagation through time (BPTT) for training.

        Resets derivative caches and propagates errors backward through time.

        Args:
            example (object): The training example to process.
        """
        for group in self.groups:
            af.fill(group.outputderivCache, 0)
            for transform in group.output_transforms:
                if transform.name in {'Out_Integr'}:
                    transform.unitData = af.zeros(transform.group.num_units)
            for transform in group.input_transforms:
                if transform.name in {'In_Integr'}:
                    transform.unitData = af.zeros(transform.group.num_units)

        self.backward()

        for tick in range(self.ticks_on_example-2, -1, -1):
            # restore output derives, output and input
            for group in self.groups:
                if group.name == "output":
                    group.output_derivs = group.output_derivs_history[tick]
                else:
                    group.output_derivs = af.zeros(group.num_units)
                if (group.group_type != "bias"):
                    group.output_matrix = group.output_history[tick]
                    group.input_matrix = group.input_history[tick]

            self.backward()

        for group in self.groups:
            if (group.group_type != "bias"):
                group.output_matrix = group.output_history[self.ticks_on_example-1]

    def standard_net_train_tick(self, event, tick, example):
        """
        Processes a single tick of training.

        Args:
            event (object): The event being processed.
            tick (int): The current tick of the network.
            example (object): The example being trained.

        Returns:
            tuple: A tuple containing the input results, total error, and total unit cost.
        """

        # different than standard network
        input_result = []
        if event.pre_proc_name is not None:
            event.pre_proc()

        target = af.array(event.target_group[0])

        self.reset_derivs()
        group_outputs = self.forward(tick, target, example)
        output = group_outputs[-1]
        self.unit_cost, self.unit_cost_derivs = self.compute_unit_output_cost(self.output_groups)
        # check if output is greater than threshold. If so, update criterion?
        for i in range(len(output)):
            if abs(output[i] - target[i]) < self.group_criterion_threshold:
                self.group_criterion_reached = True
            else:
                self.group_criterion_reached = False

        group_outputs.append(target)
        input_result.append([s.tolist() for s in group_outputs])

        example.example_train_error.append(sum(self.errors))
        # Accumulate errors over the batch
        if self.batch_errors is None:
            self.batch_errors = self.errors
        else:
            self.batch_errors = [i + j for i,
                                 j in zip(self.batch_errors, self.errors)]
        # Accumulate unit costs over the batch
        if self.batch_unit_costs is None:
            self.batch_unit_costs = self.unit_cost
        else:
            self.batch_unit_costs = [i + j for i, j in zip(self.batch_unit_costs, self.unit_cost)]
        for i in range(len(self.output_groups)):
            self.output_groups[i].output_derivs = self.error_derivs[i] + self.unit_cost_derivs[i]
            self.output_groups[i].output_derivs_history[tick] = self.error_derivs[i] + self.unit_cost_derivs[i]

        return input_result, sum(self.errors), sum(self.unit_cost)

    def standard_net_test_tick(self, event, tick, example):
        """
        Processes a single tick of testing.

        Args:
            event (object): The event being processed.
            tick (int): The current tick of the network.
            example (object): The example being tested.

        Returns:
            list: The results of processing the input.
        """

        # different than standard network
        input_result = []
        if event.pre_proc_name is not None:
            event.pre_proc()

        target = af.array(event.target_group[0])

        self.reset_derivs()
        group_outputs = self.forward(tick, target, example)
        output = group_outputs[-1]

        # check if output is greater than threshold. If so, update criterion?
        for i in range(len(output)):
            if abs(output[i] - target[i]) < self.group_criterion_threshold:
                self.group_criterion_reached = True
            else:
                self.group_criterion_reached = False

        group_outputs.append(target)
        input_result.append([s.tolist() for s in group_outputs])
        self.test_errors, self.test_error_derivs = self.errors, self.error_derivs
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
