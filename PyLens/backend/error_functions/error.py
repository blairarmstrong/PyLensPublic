import numpy as np
from ..array_factory import Array_factory as af
from ..parameters import ErrorParamaters
from ..parameters import OptimizerParameters

optimizer_params = OptimizerParameters()

class Error(Exception):

    """
    The `Error` class provides a base for various error functions used to compute the similarity 
    between the predicted outputs and target values in a machine learning model.
    
    Parameters:
        name (str): Name of the error function.
        param (ErrorParamaters): An instance of `ErrorParamaters` containing parameters related to error handling.
        large_value (float): A large constant value used in calculations.
        small_value (float): A small constant value used in calculations.
        target_radius (float): The maximum allowed deviation between outputs and targets.
        zero_error_radius (float): The radius around the target within which no error is considered.
    """
    name = None
    param = ErrorParamaters()

    def __init__(self, name, group):
        self.name = name
        self.group = group
        self.large_value = self.param.PAR_E_large_value
        self.small_value = self.param.PAR_E_small_value
        self.target_radius = optimizer_params.PAR_O_targetRadius
        self.zero_error_radius = optimizer_params.PAR_O_zeroErrorRadius

        # Below parameters will cause issues with referencing Network
        ## Suggestion:
        ### Why don't we just pass in these values rather than referencing it from network? 
        ### Since these error functions do not modify network directly
        
        # tzr = self.group.network.optimizer.optimizer_params.PAR_O_targetZeroScaling
        # tpi = self.group.network.ticks_per_interval

        # self.target_radius = self.group.network.optimizer.optimizer_params.PAR_O_targetRadius
        # self.zero_error_radius = self.group.network.optimizer.optimizer_params.PAR_O_zeroErrorRadius

    def forward(self, outputs, targets, frequency):
        return None

    def backward(self, outputs, targets, frequency):
        return None

    # @staticmethod
    # def adjusted_target(o, t, tr, zr):
    #     if -zr < o - t < zr:
    #         adj_t = o
    #     elif o - t > tr:
    #         adj_t = t + tr
    #     elif o - t < -tr:
    #         adj_t = t - tr
    #     else:
    #         adj_t = o

    #     return adj_t

    @staticmethod
    def adjusted_target_group(outputs, targets, tr, tor, zr):
        '''
        Adjusts the target vector based on certain radii to limit the error within a given range.

        Parameters:
            outputs (ndarray): The predicted output vector.
            targets (ndarray): The target vector.
            tr (float): The target radius that limits how much the target can deviate from the output.
            tor (float): The target-one radius, applied when the target value is 1.
            zr (float): The zero-error radius, within which no error is considered.

        Returns:
            ndarray: The adjusted target vector.
        '''
        adj_t = np.full(targets.shape, np.inf)

        r = np.full(targets.shape, tr)
        if tor > 0:
            r = np.where(targets == 1, tor, r)

        adj_t = np.where((-zr < outputs - targets) & (outputs-targets < zr), outputs, adj_t)
        adj_t = np.where((outputs - targets > r) & (adj_t==np.inf), targets + r, adj_t)

        adj_t = np.where((outputs - targets < -r) & (adj_t==np.inf), targets - r, adj_t)
        adj_t = np.where(adj_t == np.inf, outputs, adj_t)
        return adj_t


        # adj_t = []
        # # temporary loop through vectors
        # for o, t in zip(outputs, targets):
        #     r = tr
        #     if t == 1 and tor > 0:
        #         r = tor
        #
        #     if -zr < o - t < zr:
        #         adj_t.append(o)
        #
        #     elif o - t > r:
        #         adj_t.append(t + r)
        #
        #     elif o - t < -r:
        #         adj_t.append(t - r)
        #
        #     else:
        #         adj_t.append(o)
        #
        #
        #
        # return af.array(adj_t)


