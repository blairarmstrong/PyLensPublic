import keyboard
import torch.nn as nn
import torch
import torch.optim as optim
from .network import Network
from .parameters import NetworkParameters
from .parameters import OptimizerParameters
from .array_factory import Array_factory as af

network_params = NetworkParameters()
optimizer_params = OptimizerParameters()

class PytorchNetwork(Network):
    """
    A PyTorch-based neural network class that extends the base `Network` class.

    This class provides functionalities for setting different update methods,
    computing costs, and updating weights using PyTorch's optimization tools.

    Attributes:
        parameters_pytorch (list): List of trainable PyTorch tensors (weights).
        error_functions (dict): Dictionary mapping error function names to PyTorch loss functions.
    """
    parameters_pytorch = []
    error_functions={"cross_entropy": nn.CrossEntropyLoss(), "mean_squared": nn.MSELoss()}

    def compute_cost(self, output_groups, target):
        """
        Computes the cost function for the given output groups and target.

        Uses PyTorch autograd to compute error derivatives for backpropagation.

        Args:
            output_groups (list): A list of output tensors from different layers.
            target (torch.Tensor): The target output tensor.

        Returns:
            tuple: A tuple containing:
                - error_groups (list of float): List of error values per output group.
                - error_derivs (list of numpy arrays): Gradients of the error function.
        """
        error_groups = []
        error_derivs = []
        for i, output in enumerate(output_groups):
            cost = self.cost_functions[i]
            error = cost(output.output_matrix, target)
            error.retain_grad()
            error.backward(retain_graph=True)
            error_groups.append(error.detach().numpy().reshape(1,).item())
            error_derivs.append(error.grad.numpy().reshape(1,))
        return (error_groups, error_derivs)

    def set_update_method(self, lr, update_method="steepest"):
        """
        Sets the weight update method for optimization.

        The available update methods are:
        - "steepest"
        - "momentum"
        - "dougs momentum"
        - "delta bar delta"
        - "adam"

        If an invalid method is provided, it defaults to the network's predefined algorithm.

        Args:
            lr (float): Learning rate for the optimizer.
            update_method (str, optional): The name of the update method. Defaults to "steepest".
        """
        update_method_list = ["steepest", "momentum", "dougs momentum", "delta bar delta", "adam"]

        self.update_method = update_method.lower()

        self.parameters_pytorch = []
        for group in reversed(self.groups):
            if group.group_type != "input":
                for i in range(len(group.incoming_links)):
                    self.parameters_pytorch.append(group.incoming_links[i].weights)

        # if weight update algorithm is not specified, use default settings
        if self.update_method not in update_method_list:
            self.update_method = network_params.PAR_N_algorithm.lower()

        if self.update_method == "steepest":
            self.optimizer = optim.SGD(self.parameters_pytorch, lr)
        elif self.update_method == "momentum":
            self.optimizer = optim.SGD(self, lr)
        elif self.update_method == "adam":
            self.optimizer = optim.Adam(self.parameters_pytorch, lr)
        else:
            self.optimizer = optim.SGD(self.parameters_pytorch, lr)

        print("initialized " + self.update_method + " optimizer\n")

    def update_weights(self, report_request=False):
        """
        Updates the weights of the network using the assigned optimizer.

        After updating, the optimizer's gradients are reset to avoid accumulation.

        Args:
            report_request (bool, optional): If True, reports post-update statistics. Defaults to False.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()
        for group in reversed(self.groups):
            if group.group_type != "input":
                group.reset_report_params()
                for i in range(len(group.incoming_links)):
                    if report_request:
                        group.post_update_report_stat(group.incoming_links[i])
                if report_request:
                    group.aggregate_report_stats(self.report_stats)

    def compute_back(self):
        pass

