.. _heal:

heal_group
===========

Restores all lesioned units in the group.

Usage
-----

.. code-block:: python

   group.heal()

Description
-----------

``heal`` **restores all previously lesioned units** in a group by setting `lesion_mask` to `None`. This ensures:

- **All units are reactivated** after being lesioned.
- The group is fully functional again in the network.

This function is useful for **recovering from lesioning experiments** and **reactivating suppressed units**.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Heal a group that was previously lesioned:**

.. code-block:: python

   group.heal()

**2. Check if a group is healed:**

.. code-block:: python

   if group.lesion_mask is None:
       print("The group is fully healed.")

See Also
--------

- :meth:`lesion_all_groups <your_module.lesion_all_groups>`: Lesions all groups in the network.
- :meth:`lesion_specific_group <your_module.lesion_specific_group>`: Applies a lesion to a specific group.
- :meth:`heal_by_proportion <your_module.heal_by_proportion>`: Restores a proportion of lesioned units.
