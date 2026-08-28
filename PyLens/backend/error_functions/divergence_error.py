from .error import Error
from ..array_factory import Array_factory as af


class DivergenceError(Error):
    """
    DivergenceError computes the Kullback-Leibler (KL) divergence between the target and output vectors:
        t * log(t/o),
    where `t` is the target and `o` is the output.
    This is the default error function for softmax output groups and is only stable if both the output and target vectors
    are normalized to sum to 1.0.
    
    """

    def __init__(self, group):
        super().__init__("Divergence Error", group)

    def forward(self, outputs, targets, frequency):
        """
        Computes the forward pass of the divergence error.

        Parameters:
            outputs (ndarray): The predicted output vector.
            targets (ndarray): The target vector.
            frequency (int): Frequency factor used to scale the error.

        Returns:
            tuple: A tuple containing:
                - total_error (float): The total divergence error over all units.
                - adjusted_targets (ndarray): The adjusted target vector (if applicable).
        """
        error_scale = self.group.error_scale

        tr = self.target_radius
        zr = self.zero_error_radius
        tpi = self.group.network.ticks_per_interval
        pef = self.group.network.pseudoExampleFreq

        if tr != 0.0 or zr != 0.0:
            targets = self.adjusted_target_group(outputs, targets, tr, 0, zr)

        unit_error = self.func(outputs, targets)

        total_error = af.sum(unit_error)
        total_error *= error_scale / tpi * (frequency if pef else 1)

        return total_error, targets
    
    def func(self, outputs, targets):
        safe_targets = af.where(targets == 0, 1.0, targets)
        safe_outputs = af.where(outputs <= 0, 1.0, outputs)

        result = af.where(
            targets == 0,
            0.0,
            af.where(
                outputs <= 0,
                targets * af.log(safe_targets * 1e8),
                targets * af.log(safe_targets / safe_outputs)
            )
        )
        return result

    def backward(self, outputs, targets, frequency):
        error_scale = self.group.error_scale
        pef = self.group.network.pseudoExampleFreq
        scale = (frequency if pef else 1) * error_scale

        safe_o = af.where(outputs <= 0, 1.0, outputs)

        derivs = af.where(
            targets == 0,
            0.0,
            af.where(
                outputs <= 0,
                -targets * 1e8,
                -targets / safe_o
            )
        )

        return scale * derivs

if __name__ == "__main__":
    pass
    # m = DivergenceError()
    # a = np.array([3, 4, 5, 6, 7])
    # b = np.array([6, 7, 8, 9, 10])
    # import time
    # start = time.time()
    # print(m.autograd_backward(a, b, 1))
    # print(time.time() - start)
    # start = time.time()
    # print(m.backward(a, b, 1))
    # print(time.time() - start)
