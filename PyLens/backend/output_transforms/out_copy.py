from .basic import Basic

class Out_Copy(Basic):
    def __init__(self, group):
        super().__init__("Out_Copy", group)
        self.source_group = None
        self.source_field = "outputs"

    def forward(self, x):
        output = self.group.output_matrix

        if self.source_group is None:
            return output

        data = self._read_group_field(
            self.source_group,
            self.source_field
        )

        if output.shape[-1] != data.shape[-1]:
            raise ValueError(
                f"copyConnect source ({data.shape[-1]}) and copy "
                f"({output.shape[-1]}) groups must have the same number of units."
            )

        output[...] = data
        return output
