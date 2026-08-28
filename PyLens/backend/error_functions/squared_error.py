from .error import Error
from ..array_factory import Array_factory as af


class SquaredError(Error):
    '''
    The `SquaredError` class represents the mean squared error (MSE) loss function, 
    which computes the squared difference between the predicted outputs and target values.
    
    Parameters:
        name (str): The name of the error function.
        x: Placeholder for potential future use.
        func_deriv (function): Derivative of the squared error function for use in the backward pass.
    '''
    name = None
    x = None

    def __init__(self, group):
        super().__init__("Squared Error", group)

    def func(self, outputs, targets):
        """
        The squared error function, which calculates the squared difference between outputs and targets.

        Parameters:
            outputs (ndarray): The predicted values.
            targets (ndarray): The target or expected values.

        Returns:
            ndarray: The squared error for each pair of output and target.
        """
        return (targets - outputs)*(targets - outputs)

    def forward(self, outputs, targets, frequency):
        """
        Computes the forward pass of the squared error by calculating the error between predicted values and targets.

        Parameters:
            outputs (ndarray)
            targets (ndarray)
            frequency (int): Frequency factor used to scale the error.

        Returns:
            tuple: A tuple containing the total error and the possibly adjusted target vector.
        """
        t = targets
        o = outputs
        error_scale = self.group.error_scale

        tr = self.target_radius
        tor = self.group.network.optimizer.optimizer_params.PAR_O_targetOneRadius
        zr = self.zero_error_radius
        tzr = self.group.network.optimizer.optimizer_params.PAR_O_targetZeroScaling
        tpi = self.group.network.ticks_per_interval
        pef = self.group.network.pseudoExampleFreq

        if tr != 0.0 or tor != 0.0 or zr != 0.0 or tzr != 1.0:

            t = self.adjusted_target_group(o, t, tr, tor, zr)
            # adj_t = t
            unit_error = self.func(o, t)

            if tzr != 1.0:
                unit_error = af.where(targets == 0, unit_error * tzr, unit_error)

        else:
            unit_error = self.func(o, t)

        total_error = af.sum(unit_error)
        total_error *= error_scale / tpi * (frequency if pef else 1)

        # print("error")
        # print(total_error)
        return total_error, t

    def backward(self, outputs, targets, frequency):
        """
        Computes the backward pass by calculating the gradient of the squared error function 
        with respect to the outputs and scaling it based on various network parameters.

        Parameters:
            outputs (ndarray)
            targets (ndarray)
            frequency (int)

        Returns:
            ndarray: The gradient of the squared error with respect to the outputs.
        """
        o = outputs
        t = targets
        tzr = self.group.network.optimizer.optimizer_params.PAR_O_targetZeroScaling
        pef = self.group.network.pseudoExampleFreq

        error_scale = self.group.error_scale
        scale = (frequency if pef else 1) * (error_scale * 2)
        zero_target_scale = scale * tzr

        deriv = o - t 
        if tzr != 1:
            mult = af.where(t == 0, zero_target_scale, scale)
            return deriv*mult

        return deriv*scale
