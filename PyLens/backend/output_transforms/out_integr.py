from .modifying import Modifying
import copy

class Out_Integr(Modifying):
    """This is just like IN_INTEGR but it integrates the output rather than the input.
     This is put on by default in a CONTINUOUS network unless IN_INTEGR is specified.
    """

    def __init__(self, group):
        super().__init__("Out_Integr", group)
        self.dt = self.group.network.dt * self.group.dt

    def forward(self):
        tick = self.group.curr_tick

        self.unitHistoryData[tick] = self.group.output_matrix

        self.unitData += self.dt * (
            self.group.output_matrix - self.unitData
        )

        self.group.output_matrix[...] = self.unitData


    def backward(self):
        lastoutputderiv = self.unitData

        self.unitData += self.dt * (self.group.output_derivs - lastoutputderiv)

        self.group.output_derivs[...] = self.unitData
        self.group.output_matrix[...] = (
            self.unitHistoryData[self.group.curr_tick]
        )

