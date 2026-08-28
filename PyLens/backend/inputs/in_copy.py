from .input_transform import Input_Transform

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

    def compute(self, prev_links):
        """
        Copies the output matrix of the incoming group(s) to the current group.

        Args:
            prev_links (list): List of links connecting to the current group.

        Returns:
            np.array: Copied output matrix from incoming groups.
        """
        if self.source_group is not None:
            return self._read_group_field(self.source_group, self.source_field)

        if prev_links:
            return prev_links[-1].outgoing_group.output_matrix
        return self.group.input_matrix * 0
    
    def backward(self, prev_links, input_derivs):
        return None
