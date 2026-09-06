from .basic import Basic
from ..array_factory import Array_factory as af


class Kohonen(Basic):
    """Kohonen
    """

    def __init__(self, group):
        super().__init__("kohonen", group)

    def distance_squared(self, ai, aj, bi, bj, cols, rows, periodic):
        di = af.absolute(ai - bi)
        dj = af.absolute(aj - bj)
        if periodic:
            di = af.where(2 * di > cols, cols - di, di)
            dj = af.where(2 * dj > rows, rows - dj, dj)

        return af.square(di) + af.square(dj)

    def forward(self):
        x = self.group.input_matrix

        cols = self.group.num_cols
        rows = self.group.num_units // cols
        periodic = False

        max_input = af.max(x)
        mu = af.argmin(x)

        scale = 1 / max_input
        neigh = self.group.neighborhood ** 2

        mi = mu % cols
        mj = mu // cols

        cols_array = af.zeros(x.shape[0]) + cols

        i = af.remainder(
            af.asarray(range(x.shape[0])),
            cols_array
        )
        j = af.asarray(range(x.shape[0])) // cols

        dist = self.distance_squared(
            i, j, mi, mj, cols, rows, periodic
        )

        self.group.output_matrix[...] = af.where(
            dist <= neigh,
            1 - x * scale,
            0
        )

    def backward(self):
        self.group.input_derivs[...] = af.where(
            self.group.output_matrix > 0,
            1,
            0
        )
