from .modifying import Modifying
from ..array_factory import Array_factory as af


class Out_Winner(Modifying):

    def __init__(self, group):
        super().__init__("Out_Winner", group)


    def forward(self, x):
        # original = self.unitHistoryData[self.group.curr_tick]
        self.unitHistoryData[self.group.curr_tick] = x
        # dim = x.shape
        min = self.group.minOutput

        max = af.amax(x)
        # change multiple max values?
        x = af.where(x == max, x, min)
        return x


    def backward(self, x, output_derivs):

        return self.unitHistoryData[self.group.curr_tick]
