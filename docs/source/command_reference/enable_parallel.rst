.. _enable_parallel:

enable_parallel
===============

Enable network training in parallel mode.

Usage
-----

.. code-block:: python

    net.parallel_mode = True
    net.num_worker = 3
    net.train()

    # or equivalently
    net.train(parallel_mode=True, num_worker=3)

Description
-----------

To enable paralle training, you can either 

1. set the property of the network to enable parallel training:

- ``net.parallel_mode = True`` tells PyLens that network training will be conducted in parallel model
- ``net.num_worker = 3`` tells PyLens to use 3 cpu cores to run the paralle training.
- Then run net.train() as usual.

2. or equivalently, you can directly start parallel training by adding ``parallel_mode=True``, ``num_worker=3`` into the arguments of ``net.train()``.

To go back to serial (non-parallel) training. Set ``net.parallel_mode = False`` or run ``net.train(parallel_mode=False)`` to directly start serial training.


See Also
--------

- :doc:`Parallel training <../parallel>`: Overview of parallel training in PyLens
