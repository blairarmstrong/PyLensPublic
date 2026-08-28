.. _reset_network:

reset_network
=============

.. the description is stored in the conf.py file

|reset_network|

Usage
-----

.. code-block:: python

   net.reset_network()

Description
-----------

``reset_network`` restores the **network state** to its initial conditions. This function resets:

- **Optimizer parameters**: Calls ``reset_weights()`` to reinitialize weights. Calls ``reset_weight_derivs()`` to reset weight derivatives.

- **Network properties**: Resets **output values** and **derivatives**. Sets ``num_update`` back to ``0``.

- **Visualization Data**: Restarts all **graph traces**.

- **Gain Parameter**: Resets ``gain`` to ``network_params.PAR_N_gain``.

This function is useful for **restarting training** or **reinitializing network behavior** without reloading from scratch.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Reset the network before retraining:**

.. code-block:: python

   net.reset_network()

**2. Reset a network after modifying optimizer parameters:**

.. code-block:: python

   net.optimizer.learning_rate = 0.01
   net.reset_network()

See Also
--------

- :doc:`reset_derivs`: |reset_derivs|
- :doc:`reset_unit_values`: |reset_unit_values|
- :doc:`reset_example_list`: |reset_example_list|
