import numpy as np
import torch
import os

class Array_factory:
    baseType = None

    # backend-independent constants usable before configuration
    nan = float("nan")
    NaN = float("nan")

    @classmethod
    def set_base_type(cls, baseType):
        cls.baseType = baseType

        if baseType == 'numpy':
            cls.float32 = np.float32
            cls.float64 = np.float64
            cls.int32 = np.int32
            cls.int64 = np.int64
            cls.bool = np.bool_
            cls.newaxis = np.newaxis
            cls.ndarray = np.ndarray

            cls.array = staticmethod(np.array)
            cls.zeros = staticmethod(np.zeros)
            cls.zeros_like = staticmethod(np.zeros_like)
            cls.ones = staticmethod(np.ones)
            cls.ones_like = staticmethod(np.ones_like)
            cls.empty = staticmethod(np.empty)

            cls.exp = staticmethod(np.exp)
            cls.log = staticmethod(np.log)
            cls.sum = staticmethod(np.sum)
            cls.sqrt = staticmethod(np.sqrt)
            cls.diag = staticmethod(np.diag)
            cls.dot = staticmethod(np.dot)
            cls.norm = staticmethod(np.linalg.norm)
            cls.tanh = staticmethod(np.tanh)

            cls.multiply = staticmethod(np.multiply)
            cls.greater = staticmethod(np.greater)
            cls.less = staticmethod(np.less)
            cls.logical_xor = staticmethod(np.logical_xor)
            cls.maximum = staticmethod(np.maximum)
            cls.minimum = staticmethod(np.minimum)
            cls.power = staticmethod(np.power)

            cls.where = staticmethod(np.where)
            cls.isnan = staticmethod(np.isnan)
            cls.clip = staticmethod(np.clip)
            cls.square = staticmethod(np.square)
            cls.amax = staticmethod(np.amax)
            cls.count_nonzero = staticmethod(np.count_nonzero)

            cls.asarray = staticmethod(np.asarray)
            cls.ravel = staticmethod(np.ravel)
            cls.append = staticmethod(np.append)
            cls.copy = staticmethod(np.copy)
            cls.max = staticmethod(np.max)
            cls.size = staticmethod(np.size)

            cls.random_uniform = staticmethod(np.random.uniform)
            cls.random_normal = staticmethod(np.random.normal)
            cls.random_choice = staticmethod(np.random.choice)
            cls.fill_diagonal = staticmethod(np.fill_diagonal)

            cls.fill = staticmethod(
                lambda a, value: a.fill(value)
            )

            cls.astype = staticmethod(
                lambda a, dtype: a.astype(dtype)
            )

        elif baseType == 'pytorch':
            torch.set_default_dtype(torch.float64)
            cls.float32 = torch.float32
            cls.float64 = torch.float64
            cls.int32 = torch.int32
            cls.int64 = torch.int64
            cls.bool = torch.bool
            cls.newaxis = None
            cls.ndarray = torch.Tensor

            cls.zeros = staticmethod(torch.zeros)
            cls.zeros_like = staticmethod(torch.zeros_like)
            cls.ones = staticmethod(torch.ones)
            cls.ones_like = staticmethod(torch.ones_like)
            cls.empty = staticmethod(torch.empty)

            cls.exp = staticmethod(torch.exp)
            cls.log = staticmethod(torch.log)
            cls.sum = staticmethod(torch.sum)
            cls.diag = staticmethod(torch.diag)
            cls.dot = staticmethod(torch.matmul)
            cls.norm = staticmethod(torch.linalg.norm)
            cls.tanh = staticmethod(torch.tanh)

            cls.multiply = staticmethod(torch.mul)
            cls.greater = staticmethod(torch.gt)
            cls.less = staticmethod(torch.lt)
            cls.logical_xor = staticmethod(torch.logical_xor)
            cls.maximum = staticmethod(torch.maximum)
            cls.minimum = staticmethod(torch.minimum)

            cls.where = staticmethod(torch.where)
            cls.clip = staticmethod(torch.clamp)
            cls.square = staticmethod(torch.square)
            cls.amax = staticmethod(torch.amax)
            cls.count_nonzero = staticmethod(torch.count_nonzero)

            cls.asarray = staticmethod(torch.as_tensor)
            cls.ravel = staticmethod(torch.ravel)
            cls.copy = staticmethod(torch.clone)
            cls.max = staticmethod(torch.max)
            cls.size = staticmethod(torch.numel)

            def sqrt(x):
                if torch.is_tensor(x):
                    return torch.sqrt(x)
                return x ** 0.5

            def power(a, b):
                if not torch.is_tensor(a) and not torch.is_tensor(b):
                    return a ** b
                return torch.pow(a, b)

            def isnan(x):
                return torch.isnan(torch.as_tensor(x))

            def random_uniform(low=0.0, high=1.0, size=None):
                return torch.empty(size).uniform_(low, high)

            def random_normal(loc=0.0, scale=1.0, size=None):
                return torch.normal(mean=loc, std=scale, size=size)

            def fill_diagonal(a, val):
                return a.fill_diagonal_(val)

            def append(arr, values, axis=None):
                arr = torch.as_tensor(arr)
                values = torch.as_tensor(
                    values,
                    dtype=arr.dtype,
                    device=arr.device
                )

                if axis is None:
                    return torch.cat(
                        (torch.ravel(arr), torch.ravel(values))
                    )

                return torch.cat((arr, values), dim=axis)

            def fill(a, value):
                a.fill_(value)

            def astype(a, dtype):
                if dtype is int:
                    dtype = torch.int64
                elif dtype is float:
                    dtype = torch.float64

                return a.to(dtype=dtype)

            def array(x, *args, **kwargs):
                if torch.is_tensor(x):
                    return x.clone()

                if isinstance(x, (list, tuple)) and any(torch.is_tensor(v) for v in x):
                    return torch.stack([
                        v if torch.is_tensor(v) else torch.as_tensor(v)
                        for v in x
                    ])

                return torch.tensor(x, *args, **kwargs)

            cls.array = staticmethod(array)
            cls.sqrt = staticmethod(sqrt)
            cls.power = staticmethod(power)
            cls.isnan = staticmethod(isnan)
            cls.random_uniform = staticmethod(random_uniform)
            cls.random_normal = staticmethod(random_normal)
            cls.fill_diagonal = staticmethod(fill_diagonal)
            cls.append = staticmethod(append)
            cls.fill = staticmethod(fill)
            cls.astype = staticmethod(astype)

        else:
            raise ValueError(
                f"Unknown BASETYPE {baseType!r}; "
                "expected 'numpy' or 'pytorch'."
            )
