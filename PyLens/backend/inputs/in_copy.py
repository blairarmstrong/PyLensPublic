from .input_transform import Input_Transform
from ..array_factory import Array_factory as af

class In_Copy(Input_Transform):
    """
    Copies the output of incoming groups to the current group.
    """
    def __init__(self, group):
        """
        Initializes the In_Copy transformation.

        Args:
            group: The neural network group to which this transformation is applied.
        """
        super().__init__("In_Copy", group)
        self.source_group = None
        self.source_field = "outputs"

    def forward(self):
        """
        Copies the output matrix of the incoming group(s) to the current group.
        """
        if self.source_group is not None:
            self.group.input_matrix[...] = self._read_group_field(
                self.source_group,
                self.source_field
            )
        elif self.group.incoming_links:
            self.group.input_matrix[...] = (
                self.group.incoming_links[-1]
                .outgoing_group.output_matrix
            )
        else:
            af.fill(self.group.input_matrix, 0)
    
    def backward(self):
        pass
