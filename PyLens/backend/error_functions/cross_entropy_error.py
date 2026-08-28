from ..array_factory import Array_factory as af
from PyLens.backend.error_functions.error import Error


class CrossEntropyError(Error):
    """
    CrossEntropyError is a class for computing the cross-entropy error, commonly used in classification tasks.
    The cross-entropy error is computed as:
        t * log(t/o) + (1 - t) * log((1 - t) / (1 - o)),
    where `t` is the target, and `o` is the output.

    This can become infinite if the output incorrectly reaches 0.0 or 1.0.
    This may happen if the training parameters are too aggressive.
    Lens caps the error at a very large value.
    CROSS_ENTROPY is the default error type for most output groups.
    
    """
    def __init__(self, group):
        super().__init__("Cross Entropy Error", group)

    def forward(self, outputs, targets, frequency):
        """
        Computes the forward pass of the cross-entropy error.

        Parameters:
            outputs (ndarray): The predicted output vector.
            targets (ndarray): The target vector.
            frequency (int): Frequency factor used to scale the error.

        Returns:
            tuple: A tuple containing:
                - total_error (float): The total cross-entropy error over all units.
                - adjusted targets (ndarray): The adjusted target vector (if applicable).
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
            unit_error = self.func(o, t)

            if tzr != 1.0:
                unit_error = af.where(unit_error == 0.0, unit_error*tzr, unit_error)

        else:
            unit_error = self.func(o, t)

        total_error = af.sum(unit_error)
        total_error *= error_scale/tpi * (frequency if pef else 1)

        return total_error, t

    def func(self, outputs: af, targets: af):
        '''
        Cross-entropy function for a batch of outputs and targets.
        
        Parameters:
            outputs (ndarray)
            targets (ndarray)
        
        Returns:
            ndarray: Cross-entropy error values for each unit.
        '''

        # target == 0
        valid = (targets == 0.0) & (outputs != 1.0)
        safe_outputs = af.where(valid, outputs, 0.0)

        error_zero = af.where(
            outputs == 1.0,
            self.large_value,
            -af.log(1.0 - safe_outputs)
        )

        # target == 1
        valid = (targets == 1.0)
        safe_outputs = af.where(valid, outputs, 1.0)

        error_one = af.where(
            outputs == 0.0,
            self.large_value,
            -af.log(safe_outputs)
        )

        # target != 0 and target != 1
        valid = (
            (targets != 0.0)
            & (targets != 1.0)
            & (outputs > 0.0)
            & (outputs < 1.0)
        )

        safe_outputs = af.where(valid, outputs, 0.5)
        safe_targets = af.where(valid, targets, 0.5)

        error_other = af.where(
            (outputs <= 0.0) | (outputs >= 1.0),
            self.large_value,
            safe_targets * af.log(safe_targets / safe_outputs)
            + (1.0 - safe_targets) * af.log(
                (1.0 - safe_targets) / (1.0 - safe_outputs)
            )
        )

        unit_error = af.where(
            targets == 0.0,
            error_zero,
            af.where(
                targets == 1.0,
                error_one,
                error_other
            )
        )

        return unit_error

    # def deriv_vec(self, o, t):
    #     '''
    #     Vectorized helper function to calculate the derivative of the cross-entropy error.
        
    #     Parameters:
    #         o (float)
    #         t (float)
        
    #     Returns:
    #         float: Derivative of the cross-entropy error for the given output and target.
    #     '''
    #     if t == 0:
    #         return self.large_value if 1-o<=self.small_value else 1/(1-o)
    #     elif t == 1:
    #         return -self.large_value if o <= self.small_value else -1/o
    #     else:
    #         return (o-t)*self.large_value if o*(1-o) <= self.small_value else (o-t)/(o*(1-o))

    def backward(self, outputs, targets, frequency):
        target = targets

        error_scale = self.group.error_scale
        pef = self.group.network.pseudoExampleFreq
        target_zero_scaling = self.group.network.optimizer.optimizer_params.PAR_O_targetZeroScaling

        scale = (frequency if pef else 1) * error_scale
        zero_target_scale = scale * target_zero_scaling

        # target == 0
        large_deriv = (1.0 - outputs) <= self.small_value
        safe_outputs = af.where(
            large_deriv,
            0.0,
            outputs
        )

        deriv_zero = af.where(
            large_deriv,
            self.large_value,
            1.0 / (1.0 - safe_outputs)
        )

        # target == 1
        large_deriv = outputs <= self.small_value
        safe_outputs = af.where(
            large_deriv,
            1.0,
            outputs
        )

        deriv_one = af.where(
            large_deriv,
            -self.large_value,
            -1.0 / safe_outputs
        )

        # target != 0 and target != 1
        denom = outputs * (1.0 - outputs)
        large_deriv = denom <= self.small_value
        safe_denom = af.where(
            large_deriv,
            1.0,
            denom
        )

        deriv_other = af.where(
            large_deriv,
            (outputs - target) * self.large_value,
            (outputs - target) / safe_denom
        )

        deriv = af.where(
            target == 0.0,
            deriv_zero,
            af.where(
                target == 1.0,
                deriv_one,
                deriv_other
            )
        )

        if target_zero_scaling != 1.0:
            mult = af.where(
                target == 0.0,
                zero_target_scale,
                scale
            )
            return deriv * mult

        return deriv * scale

if __name__ == "__main__":
    pass
