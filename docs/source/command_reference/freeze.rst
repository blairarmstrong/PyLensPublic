.. _freeze:

freeze
======

Freezes the weight updates for the link.

Usage
-----

.. code-block:: python

   link.freeze()

Description
-----------

``freeze`` prevents further **weight updates** for the link by setting its **`frozen`** attribute to `True`. This ensures that:

- **Weights remain unchanged** during training or optimization.
- The link retains its current values **without being updated** by gradient-based methods.

This function is useful when **certain connections need to be fixed** while other parts of the network continue to learn.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Freeze a specific link to prevent weight updates:**

.. code-block:: python

   link.freeze()

**2. Verify if a link is frozen:**

.. code-block:: python

   if link.frozen:
       print("This link is frozen and will not update.")

See Also
--------

- :meth:`unfreeze <your_module.unfreeze>`: Unfreezes the link to allow weight updates.
- :meth:`update_weight <your_module.update_weight>`: Updates the link's weight if it is not frozen.
- :meth:`train <your_module.train>`: Runs training, which updates unfrozen weights.
