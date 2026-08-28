from .modifying import Modifying
import copy

class Out_Integr(Modifying):
    """This is just like IN_INTEGR but it integrates the output rather than the input.
     This is put on by default in a CONTINUOUS network unless IN_INTEGR is specified.
    """

    def __init__(self, group):
        super().__init__("Out_Integr", group)
        self.dt = self.group.network.dt * self.group.dt


    def func(self, x):
        lastoutput = self.unitData
        if self.group.network.parallel_mode:
            self.unitHistoryData = self.unitHistoryData.copy()
        self.unitHistoryData[self.group.curr_tick] = x
        self.unitData += self.dt * (x - lastoutput)

        return self.unitData

    def forward(self, x):
        return self.func(x)

    def backward(self, x, output_derivs):
        lastoutputderiv = self.unitData

        self.unitData += self.dt * (output_derivs - lastoutputderiv)

        self.group.output_derivs = copy.copy(self.unitData)
        self.group.output_matrix = copy.copy(self.unitHistoryData[self.group.curr_tick])
        return self.unitHistoryData[self.group.curr_tick]
