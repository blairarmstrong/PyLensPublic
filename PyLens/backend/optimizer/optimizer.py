import numpy as np
from ..parameters import OptimizerParameters


class Optimizer:
    """
    Optimizer used to perform gradient descent for the training of neural network. It serves as the base for different optimizing algorithm.
    """

    def __init__(self, network, lr):
        """
        Args:
            network (Network): The neural network whose weights will be managed/updated.
            lr (float or None): Optional override for the default learning rate. If None, 
                                uses the value from `OptimizerParameters`.
        """
        self.optimizer_params = OptimizerParameters()
        self.network = network
        self.weights_list = []
        self.weight_decay = self.optimizer_params.PAR_O_weightDecay
        self.weight_elimination = self.optimizer_params.PAR_O_weightEliminationW0
        if lr is None:
            self.learning_rate = self.optimizer_params.PAR_O_learningRate
        else:
            self.learning_rate = lr
        self.momentum = self.optimizer_params.PAR_O_momentum
        self.groups = network.groups
        self.rate_increment = self.optimizer_params.PAR_O_rate_increment
        self.rate_decrement = self.optimizer_params.PAR_O_rate_decrement

    def update_weights(self, report_request=None):
        """
        Updates the weights of the network.

        This method is intended to be overridden by subclasses implementing specific 
        optimization algorithms. 

        Args:
            report_request (bool, optional): If True, triggers reporting/statistics 
                                             during the update step.
        """
        pass

    def reset_weight_derivs(self):
        """
        Resets the weight derivatives (gradients) for all incoming links in all groups.
        """
        for group in self.network.groups:
            for link in group.incoming_links:
                link.reset_weight_derivs()

    def reset_weights(self):
        """
        Resets the weights of all incoming links in all groups, using the network's 
        specified random range (`network.rand_range`).
        """
        for group in self.network.groups:
            for link in group.incoming_links:
                link.reset_weights(rand_range=self.network.rand_range)
