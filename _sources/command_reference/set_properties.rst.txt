.. _set_properties:

set_properties
==============

In Python, you can directly assign values to network properties, such as ``net.name = "first_network"``. However, properties set this way won’t be properly received by parallel workers. Instead, use ``set_properties()`` to set network properties so they can be correctly transmitted to parallel workers. This is required for parallel training.

Usage
-----

.. code-block:: python

    net.set_properties(**updates)

Description
-----------

This function sets the properties of the network so that parallel training can be properly run. For details, see :doc:`../parallel`.

Arguments
---------

- **\*\*updates**: Property assignments to be executed. For example, ``name='first_network'`` sets the network name to ``first_network``. This is equivalent to ``net.name = 'first_network'``, but it enables proper setup for parallel training.

Examples
--------

**1. Set the properties of a Boltzmann digit network that supports parallel training:**

.. code-block:: python

    # set network and example properties
    net.set_properties(
        group_criterion_threshold=0.001,
        test_group_criterion_threshold=0.001,
        clamp_strength=1.0,
        init_gain=0.1,
        final_gain=1.0,
        anneal_time=1.0,
        training_sets=[
            {"max_time": 3.0, "min_time": 0.0, "grace_time": 1.0}
        ]
    )

This ``set_properties()`` call is equivalent to running the following lines in non-parallel mode, **but it is required to enables proper parallel training.**

.. code-block:: python

    # set network properties
    net.group_criterion_threshold = 0.001
    net.test_group_criterion_threshold = 0.001
    net.clamp_strength = 1.0
    net.init_gain = 0.1
    net.final_gain = 1.0
    net.anneal_time = 1.0

    # set example set properties
    net.training_sets[0].max_time = 3.0
    net.training_sets[0].min_time = 0.0
    net.training_sets[0].grace_time = 1.0

See Also
--------

- :doc:`train`: |train|
