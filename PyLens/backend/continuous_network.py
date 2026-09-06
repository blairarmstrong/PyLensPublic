from .network import Network
from .parameters import NetworkParameters
from .parameters import OptimizerParameters
from .array_factory import Array_factory as af

network_params = NetworkParameters()
optimizer_params = OptimizerParameters()


class ContinuousNetwork(Network):
    """
    This class implements a continuous-time neural network. It runs and trains a continuous neural network over multiple time ticks.
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

        """
        Attributes:
            network_type (str): Specifies the network type as 'continuous'.
            max_ticks (int): Maximum number of ticks in a full run.
        """
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

        self.network_type = 'continuous'
        self.max_ticks = self.ticks_per_interval * self.time_intervals + 1

    def forward(self, tick):

        """
        Performs a forward pass through the network.

        Args:
            tick (int): The current tick of the network.

        """

        # compute input for all groups
        for group in self.groups:
            group.curr_tick = tick
            if group.group_type != "bias":
                group.compute_input()
                group.input_history[tick] = group.input_matrix

        # compute output for all groups
        for group in self.groups:
            if group.group_type != "bias":
                group.compute_output()

                if group.lesion_mask is not None:
                    group.output_matrix *= group.lesion_mask

                # reinitialize dropout mask
                group.unit_dropout(group.dropout_rate)

                if group.dropout_mask is not None:
                    group.output_matrix *= group.dropout_mask

                group.input_set = False
                group.output_history[tick] = group.output_matrix

    def net_train_example_back(self):
        """
        Performs backpropagation through time (BPTT) for training.

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

        for tick in range(self.ticks_on_example-1, 0, -1):
            # When you get here, the outputs are the outputs from tick and the
            # outputDeriv caches contain the backpropagated error from the
            # next tick.  

            # restore output_derivs, Set outputDerivs to the
            # stored instant error derivatives. 
            for group in self.groups: 
                if group in self.output_groups:
                    group.output_derivs[...] = group.output_derivs_history[tick]
                else:
                    af.fill(group.output_derivs, 0)

            # Compute output backward
            for group in self.groups:
                group.curr_tick = tick
                group.compute_output_back()

                if group.group_type != "elman":
                    if group.lesion_mask is not None:
                        group.input_derivs *= group.lesion_mask
                    if group.dropout_mask is not None:
                        group.input_derivs *= group.dropout_mask

            # Restore the outputs from the previous tick
            for group in self.groups:
                if (group.group_type != "bias"):
                    group.output_matrix[...] = group.output_history[tick - 1]
                    group.input_matrix[...] = group.input_history[tick - 1]

            # compute input backwards
            for group in self.groups:
                group.compute_input_back()

        for group in self.groups:
            if (group.group_type != "bias"):
                group.output_matrix[...] = group.output_history[self.ticks_on_example-1]

