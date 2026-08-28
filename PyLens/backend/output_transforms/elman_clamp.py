from ..clamping import Clamping

class Elman_Clamp(Clamping):

    def __init__(self, group):
        super().__init__("Elman_Clamp", group)
        self.source_group = None

    def forward(self, x):
        # self.group.input_matrix = x
        if self.source_group is None:
            self.source_group = self.group.incoming_links[0].outgoing_group
        elman_input = self.source_group.output_matrix_cache
        return elman_input  # context group's output is zeroed

    def backward(self, x, output_derivs):
        if self.source_group is None:
            self.source_group = self.group.incoming_links[0].outgoing_group
        elman_deriv = self.source_group.outputderivCache
        elman_input = self.source_group.output_matrix_cache

        elman_deriv += output_derivs
        self.group.output_matrix -= elman_input

        # elman_clamp does not return input_derivs
