from .basic import Basic
import numpy as np


class Kohonen(Basic):
    """Kohonen
    """

    def __init__(self, group):
        super().__init__("kohonen", group)

    def distance_squared(self, ai, aj, bi, bj, cols, rows, periodic):
        di = np.absolute(ai - bi)
        dj = np.absolute(aj - bj)
        if periodic:
            di = np.where(2 * di > cols, cols - di, di)
            dj = np.where(2 * dj > cols, rows - dj, dj)

        return np.square(di) + np.square(dj)

    def forward(self, x):
        # max = 0
        # min = float('inf')
        # i, j, mu = 0
        mi= self.group.num_cols
        mj=self.group.num_cols
        cols = self.group.num_cols
        rows = self.group.num_units//cols
        # periodic = self.group.periodicBoundary
        periodic = False

        print("x: ", x)
        max = np.max(x)
        # print("max: ", max)
        # min = np.min(x)
        mu = np.argmin(x)
        # print("mu: ", mu)
        scale = 1/max
        # print("scale: ", scale)
        neigh = self.group.neighborhood**2
        # print("neigh: ", neigh)
        mi = mu % cols
        # print("mi: ", mi)
        mj = mu / cols
        # print("mj: ", mj)
        # print(x.shape)
        cols_array = np.zeros(x.shape[0]) + cols
        # print("cols_array: ", cols_array)
        # print(cols)
        # print(cols_array)
        i = np.remainder(np.asarray(range(x.shape[0])), cols_array)
        # print("i: ", i)
        j = np.asarray(range(x.shape[0])) // cols
        # print("j: ", j)
        # print(range(x.shape[0]))
        # print(j)
        dist = self.distance_squared(i, j, mi, mj, cols, rows, periodic)

        output = np.where(dist <= neigh, 1-x*scale, 0)

        return output

    def backward(self, x, output_derivs):
        # Input deriv = 1 if output > 0, else 0
        print(x)
        input_deriv = np.where(x > 0, 1, 0)
        # self.group.input_deriv = input_deriv
        print("after: ", input_deriv)
        return np.where(x > 0, 1, 0)
