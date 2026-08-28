from .modifying import Modifying
import numpy as np
from ..array_factory import Array_factory as af
import copy


class Out_Norm(Modifying):
    """This crops the output to within the range [minOutput, maxOutput]. You may want to use this after OUT_NOISE to
     prevent outputs outside of this range.
    """

    func_deriv = None


    def __init__(self, group):
        super().__init__("Out_Norm", group)
        self.maxOutput = self.group.maxOutput
        self.minOutput = self.group.minOutput
        # Used in out_norm
        self.groupHistoryData = af.empty(((self.group.time_intervals * self.group.ticks_per_interval)+1))
        # self.unitHistoryData = af.empty(((group.time_intervals * group.ticks_per_interval)+1, group.num_units))

    def func(self, x):
        scale = af.sum(x)
        self.groupHistoryData[self.group.curr_tick] = scale
        if scale != 0:
            return x * scale
        return x

    def forward(self, x):
        tick = self.group.curr_tick
        # original_output = self.group.unitHistoryData[tick]
        scale = af.sum(x)
        self.groupHistoryData[tick] = scale
        self.unitHistoryData[tick] = x
        if scale != 0:
            scale = 1/scale
            self.groupHistoryData[tick] = scale
            return x * scale
        self.groupHistoryData[tick] = scale
        return x
        # output = self.group.output_history[tick]

        # return self.func(output)

    def backward(self, x, output_derivs):
        tick = self.group.curr_tick
        shift = af.sum(x * output_derivs)
        # original_output = self.unitHistoryData[tick]
        scale = self.groupHistoryData[tick]
        self.group.output_derivs = scale * (self.group.output_derivs - shift)
        self.group.output_matrix = copy.copy(self.unitHistoryData[self.group.curr_tick])
        return af.zeros(x.shape)
        # return self.unitHistoryData[tick]


