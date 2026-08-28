.. _reset_derivs:

reset_derivs
==============

.. the description is stored in the conf.py file

|reset_derivs|

Usage
-----

.. code-block:: python

   net.reset_derivs()

Description
-----------

``reset_derivs`` clears **all stored derivative values** across all groups in the network. This function ensures that:

- **All groups reset their derivatives** by calling ``clear_derivs()`` on each group.
- The network starts fresh for **a new round of gradient calculations**.

This function is useful for **resetting gradients before backpropagation** to prevent accumulation of old values.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Reset all derivatives before a new training step:**

.. code-block:: python

   net.reset_derivs()

See Also
--------

- :doc:`reset_network`: |reset_network|