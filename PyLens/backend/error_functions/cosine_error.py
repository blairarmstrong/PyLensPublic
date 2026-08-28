from .error import Error
from ..array_factory import Array_factory as af


class CosineError(Error):
    """
    CosineError is a class for calculating the cosine similarity between two vectors.
    Specifically, it computes 1.0 minus the cosine of the angle between the output and target vectors.
    This can be used for training and evaluation in neural networks.
    
    Note:
        - Training with this error function may be tricky, as the cost encourages correct angles
          between vectors but does not enforce the magnitude of the outputs.
        - Consider pairing with a unit cost function, such as LOGISTIC_COST, to enforce binary outputs.

    Parameters:
        name (str): The name of the error function (inherited).
        x (Any): Placeholder for intermediate data (inherited).
        func_deriv (function): Gradient of the cosine error function.
        error_scale (float): Scaling factor for the error.
        frequency (int): Frequency for scaling the error.
    """

    name = None
    x = None
    error_scale = 1
    frequency = 1

    def __init__(self, group):
        super().__init__("Cosine Error", group)

    def forward(self, outputs, targets, frequency):
        """
        Computes the forward pass of the CosineError.

        Parameters:
            outputs (ndarray): The predicted output vector.
            targets (ndarray): The target vector.
            frequency (int): Frequency factor used to scale the error.

        Returns:
            tuple: A tuple containing:
                - error (float): The calculated cosine error.
                - adjusted targets (ndarray): The adjusted target vector (if applicable).
        """
        self.frequency = frequency
        t = targets
        o = outputs

        self.error_scale = self.group.error_scale
        tr = self.target_radius
        zr = self.zero_error_radius
        tpi = self.group.network.ticks_per_interval
        pef = self.group.network.pseudoExampleFreq

        if tr != 0.0 or zr != 0.0:
            t = self.adjusted_target_group(o, t, tr, 0, zr)

        cosine = self.func(o, t)

        error = cosine * self.error_scale / tpi * (
            frequency if pef else 1)

        error *= self.error_scale / tpi * (
            frequency if pef else 1)

        return error, t

    # def scale(self, cosine, error_scale, frequency):
    #     cosine *= (frequency if self.network.pseudoExampleFreq else 1) * error_scale
    #
    #     error = (1 - cosine) * error_scale / self.network.ticks_per_interval * (
    #         frequency if self.network.pseudoExampleFreq else 1)
    #
    #     error *= error_scale / self.network.ticks_per_interval * (
    #         frequency if self.network.pseudoExampleFreq else 1)
    #
    #     return error

    def func(self, outputs, targets):
        """
        Cosine similarity function.

        Computes the cosine similarity between the output and target vectors using the formula:
        cos(theta) = (outputs dot targets) / (|outputs| * |targets|)
        
        Parameters:
            outputs (ndarray)
            targets (ndarray)
        
        Returns:
            float: 1.0 - cosine similarity between the output and target vectors.
        """
        # TODO do we need to iterate and sum?
        if (af.norm(outputs) * af.norm(targets)) == 0:
            cosine = 0
        else:
            cosine = af.dot(outputs, targets) / af.norm(outputs) * af.norm(targets)

        cosine *= (self.frequency if self.group.network.pseudoExampleFreq else 1) * self.error_scale

        return 1-cosine

    def backward(self, outputs, targets, frequency):
        '''
        Computes the backward pass (gradient) of the CosineError.
        
        Parameters:
            outputs (ndarray)
            targets (ndarray)
            frequency (int)
            
        Returns:
            ndarray: Gradient of the cosine error with respect to the outputs.
        '''
        if (af.norm(outputs) * af.norm(targets)) == 0:
            cosine = 0
        else:
            cosine = af.dot(outputs, targets) / af.norm(outputs) * af.norm(targets)
        invdotprod = 1 / af.dot(outputs, targets)
        invsqoutlen = 1 / af.sum(af.square(outputs))
        if cosine == 0.0:
            return af.zeros_like(outputs)
        return cosine * (outputs * invsqoutlen - targets * invdotprod)


if __name__ == "__main__":
    pass
    # import time
    # m = CosineError()
    # a = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    # start = time.time()
    # print(m.autograd_backward(a, b, 1))
    # print((time.time()-start)*1000)
    # start = time.time()
    # print(m.backward(a, b, 1))
    # print((time.time()-start)*1000)
