from .modifying import Modifying
from ..array_factory import Array_factory as af


class Out_Winner(Modifying):

    def __init__(self, group):
        super().__init__("Out_Winner", group)


    def forward(self):
        self.unitHistoryData[self.group.curr_tick] = (
            self.group.output_matrix
        )

        min_output = self.group.minOutput
        max_output = af.amax(self.group.output_matrix)

        self.group.output_matrix[...] = af.where(
            self.group.output_matrix == max_output,
            self.group.output_matrix,
            min_output
        )


    def backward(self):
        self.group.output_matrix[...] = (
            self.unitHistoryData[self.group.curr_tick]
        )
