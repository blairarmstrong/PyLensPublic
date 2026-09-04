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

    # name = None
    # x = None
    # error_scale = 1
    # frequency = 1

    def __init__(self, group):
        super().__init__("Cosine Error", group)

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
        out_norm = af.norm(outputs)
        tar_norm = af.norm(targets)

        if out_norm * tar_norm == 0:
            return 0.0

        return af.dot(outputs, targets) / (out_norm * tar_norm)

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
        t = targets
        o = outputs

        error_scale = self.group.error_scale
        tr = self.target_radius
        zr = self.zero_error_radius
        tpi = self.group.network.ticks_per_interval
        pef = self.group.network.pseudoExampleFreq

        if tr != 0.0 or zr != 0.0:
            t = self.adjusted_target_group(o, t, tr, 0, zr)

        freq_scale = frequency if pef else 1.0

        cosine = self.func(o, t)
        cosine *= freq_scale * error_scale

        error = (
            (1.0 - cosine)
            * error_scale
            / tpi
            * freq_scale
        )
        return error, t



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
        error_scale = self.group.error_scale
        pef = self.group.network.pseudoExampleFreq
        freq_scale = frequency if pef else 1.0

        cosine = self.func(outputs, targets)

        # Same scaled cosine stored in CLens CosineData
        cosine *= freq_scale * error_scale

        if cosine == 0.0:
            return af.zeros_like(outputs)

        dotprod = af.dot(outputs, targets)
        sqoutlen = af.sum(af.square(outputs))

        invdotprod = 1.0 / dotprod
        invsqoutlen = 1.0 / sqoutlen

        return cosine * (
            outputs * invsqoutlen
            - targets * invdotprod
        )

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
