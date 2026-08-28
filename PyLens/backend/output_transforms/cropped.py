from .modifying import Modifying
from ..array_factory import Array_factory as af
import copy


class Cropped(Modifying):
    """This crops the output to within the range [minOutput, maxOutput]. You may want to use this after OUT_NOISE to
     prevent outputs outside of this range.
    """

    func_deriv = None


    def __init__(self, group):
        super().__init__("Cropped", group)
        self.maxOutput = self.group.maxOutput
        self.minOutput = self.group.minOutput

    def func(self, x):
        x = af.where(x > self.maxOutput, self.maxOutput, x)
        x = af.where(x < self.minOutput, self.minOutput, x)
        return x

    def forward(self, x):
        self.unitHistoryData[self.group.curr_tick] = copy.copy(x)
        return self.func(x)

    def backward(self, x, y):
        return self.unitHistoryData[self.group.curr_tick]
