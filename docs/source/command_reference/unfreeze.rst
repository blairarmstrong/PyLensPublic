.. _unfreeze:

unfreeze
========

Unfreezes the weight updates for the link.

Usage
-----

.. code-block:: python

   link.unfreeze()

Description
-----------

``unfreeze`` **re-enables weight updates** for the link by setting its **`frozen`** attribute to `False`. This allows:

- **Weights to be updated** during training or optimization.
- The link to participate in gradient-based learning again.

This function is useful when **previously frozen connections need to be adjusted** in later stages of training.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Unfreeze a link to allow weight updates:**

.. code-block:: python

   link.unfreeze()

**2. Check if a link is frozen before unfreezing it:**

.. code-block:: python

   if link.frozen:
       link.unfreeze()
       print("The link is now trainable.")

See Also
--------

- :meth:`freeze <your_module.freeze>`: Freezes the link to prevent weight updates.
- :meth:`update_weight <your_module.update_weight>`: Updates the link's weight if it is not frozen.
- :meth:`train <your_module.train>`: Runs training, which updates unfrozen weights.
