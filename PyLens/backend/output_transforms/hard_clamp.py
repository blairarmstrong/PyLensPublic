from ..clamping import Clamping
from ..array_factory import Array_factory as af

class Hard_Clamp(Clamping):
    """
    If the externalInput is a real number, this sets the output to the externalInput. Otherwise it does nothing.
    """

    def __init__(self, group):
        super().__init__("Hard_clamp", group)

    def forward(self):
        self.unitHistoryData[self.group.curr_tick] = (
            self.group.external_input
        )

        self.group.output_matrix[...] = af.where(
            af.isnan(self.group.external_input),
            self.group.output_matrix,
            self.group.external_input
        )
        

    def backward(self):
        clamped = ~af.isnan(self.unitHistoryData[self.group.curr_tick])
        self.group.input_derivs[clamped] = 0
