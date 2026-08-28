from .transform import Transform


class Clamping(Transform):

    def __init__(self, name, group):
        super().__init__(name, group)
