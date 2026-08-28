Performance Optimization with Numba
=====================================

Introduction
------------

To enhance the performance of our neural network computations, we have integrated `Numba <https://numba.pydata.org/>`_, a Just-In-Time (JIT) compiler for Python, into several critical components of our project. Numba allows us to accelerate numerical functions by compiling Python code to optimized machine code at runtime, significantly speeding up execution without the need to rewrite code in a lower-level language.

Why Numba?
----------

Numba offers substantial performance improvements for numerical computations, especially in array operations and loops that are prevalent in our neural network algorithms. By utilizing Numba's JIT compilation, we achieve execution speeds close to those of compiled languages like C, all while maintaining the flexibility and simplicity of Python.

**Advantages**:

- Faster execution of numerical functions.

Implementation Details
----------------------

We have applied Numba's ``@njit`` decorator to key functions in several modules, including:

- ``cross_entropy_error.py``
- ``cosine_error.py``
- ``divergence_error.py``
- ``error.py``
- ``squared_error.py``
- ``link.py``
- ``sigmoid.py``
- ``dot_product.py``

These functions are integral to error calculations, activation functions, and other computationally intensive tasks within the network. By compiling these functions with Numba, we reduce the overhead associated with Python's interpreted execution, leading to faster training times and more efficient network performance.

Example
-------

The following example shows how Numba is applied in `cross_entropy_error.py`:

.. code-block:: python

    from numba import njit
    import numpy as np

    @njit
    def error_vec(o, t, large_value):
        if t == 0:
            return large_value if o == 1.0 else -np.log(1 - o)
        elif t == 1:
            return large_value if o == 0.0 else -np.log(o)
        else:
            if o <= 0.0 or o >= 1.0:
                return large_value
            else:
                return (t * np.log(t / o) + (1 - t) * np.log((1 - t) / (1 - o)))


Performance Improvement
-----------------------

By leveraging Numba, we have observed significant speedups in our network's training process. The JIT compilation minimizes the execution time of computationally intensive functions, particularly those involving loops and array manipulations.

References
----------

- `Numba Documentation <https://numba.pydata.org/numba-doc/latest/index.html>`_
- `Numba GitHub Repository <https://github.com/numba/numba>`_