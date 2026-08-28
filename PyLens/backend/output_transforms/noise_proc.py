from ..array_factory import Array_factory as af


def gaussian_noise(loc, scale, shape):
    return af.random_normal(loc, scale, shape)


def uniform_noise(low, high, shape):
    return af.random_uniform(low, high, shape)
