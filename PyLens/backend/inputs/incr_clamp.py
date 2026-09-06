from .input_transform import Input_Transform
from ..array_factory import Array_factory as af


class IncrementClamp(Input_Transform):

    def __init__(self, group):
        """
        Initializes the IncrementClamp transformation.

        Args:
            group: The neural network group to which this transformation is applied.
        """
        super().__init__("incr_clamp", group)

    def forward(self):
        """
        Applies a clamping transformation to the input based on the network's clamp strength.

        Args:
            x (np.array): Input values to be clamped.

        Returns:
            np.array: Clamped input values.
        """
        strength = (
            self.group.network.clamp_strength
            if af.isnan(self.group.clamp_strength)
            else self.group.clamp_strength
        )

        external_input = self.group.external_input

        self.group.input_matrix[...] = af.where(
            af.isnan(external_input),
            self.group.input_matrix,
            self.group.input_matrix + strength * external_input
        )

    def backward(self):
        pass
