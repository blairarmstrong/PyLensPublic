# from .network import Network
# from .continuous_network import ContinuousNetwork
# from .srbptt_network import SRBPTTNetwork
import ray
from .parameters import NetworkParameters
from .parameters import ExampleParameters
from .parameters import OptimizerParameters
import warnings
from .array_factory import Array_factory as af

network_params = NetworkParameters()
example_params = ExampleParameters()
optimizer_params = OptimizerParameters()

class ParallelBaseNetwork:
    """
    The ParallelBaseNetwork class is used as a base class for parallel neural network instances.
    It supports synchronization of weights, weight derivative resets, and distributed example processing.
    
    Args:
        name (str): Name of the network.
        time_intervals (int): Number of time intervals per training step.
        ticks_per_interval (int): Number of ticks within each time interval.
        learning_rate (float): Learning rate for the network.
        add_bias (bool): Whether to include bias terms in the network.
    """
    def __init__(self, name, time_intervals, ticks_per_interval, 
                 learning_rate, add_bias, baseType='numpy'):
        af.set_base_type(baseType)
        super().__init__(name=name, time_intervals=time_intervals, 
                         ticks_per_interval=ticks_per_interval, 
                         learning_rate=learning_rate, add_bias=add_bias)

    def sync_weights(self, packaged_weights):
        """
        Synchronizes weights from the central server to the local network instance.
        
        Args:
            packaged_weights (list): Nested list containing weight values for each group and its incoming links.
        """
        for j in range(len(self.groups)):
            for k in range(len(self.groups[j].incoming_links)):
                self.groups[j].incoming_links[k].weights = packaged_weights[j][k][0]

    def reset_weight_derivs(self):
        """
        Resets weight derivatives for all links in the network.
        """
        for group in self.groups:
            for link in group.incoming_links:
                link.reset_weight_derivs()

    def worker_function(self, example_set_index, start_index, steps, test):
        """
        Processes a batch of examples in parallel and computes training results.
        
        Args:
            example_set_index (int): Index of the example set to use.
            start_index (int): Starting index for iterating through examples.
            steps (int): Number of training steps to perform.
            test (bool): Whether to run the function in testing mode.
        
        Returns:
            tuple: Contains lists of results, training errors, weight derivatives, and unit costs.
        """
        result_list = []
        training_errors_list = []
        unit_costs_list = []

        if test:
            if self.testing_sets:
                self.example_sets = self.testing_sets
            else:
                self.example_sets = [self.testing_set]
        else:
            self.example_sets = self.training_sets
        example_set = self.example_sets[example_set_index]
        example_set.example_iterator.curr = example_set.example_iterator.iter_list[start_index]

        for i in range(steps):
            example = example_set.iterate_example() # this is a hack
            result, training_errors, unit_costs = self.standard_net_train_example(example, test)

            if self.network_type in ['continuous', 'srbptt']:
                self.net_train_example_back(example)

            result_list.append(result)
            training_errors_list.extend(training_errors)
            unit_costs_list.extend(unit_costs)
            if (network_params.PAR_N_reset_on_example):
                self.reset_matrices()

        all_weight_derivs = []
        for i in range(len(self.groups)):
            group_weight_derivs = []
            for j in range(len(self.groups[i].incoming_links)):
                group_weight_derivs.append(self.groups[i].incoming_links[j].weight_derivs)
            all_weight_derivs.append(group_weight_derivs)

        return result_list, training_errors_list, all_weight_derivs, unit_costs_list
    
    def get_attr(self, key):
        return getattr(self, key)


def create_parallel_network(network_class, name, time_intervals, ticks_per_interval, learning_rate, add_bias, baseType):
    """
    Factory function to dynamically create and instantiate a parallel network instance.
    
    Args:
        network_class (type): The base network class to extend.
        name (str): Name of the network.
        time_intervals (int)
        ticks_per_interval (int)
        learning_rate (float)
        add_bias (bool)
    
    Returns:
        ray.remote: A remote instance of the ParallelNetwork class.
    """

    ParallelNetwork = type(
        'ParallelNetwork',
        (ParallelBaseNetwork, network_class),
        {}
    )

    return ray.remote(ParallelNetwork).remote(name, time_intervals, ticks_per_interval, learning_rate, add_bias, baseType)
