.. _lesion_all_groups:

lesion_all_groups
=================

Applies a lesion to all groups in the network.

Usage
-----

.. code-block:: python

   net.lesion_all_groups(lesion_rate)

Description
-----------

``lesion_all_groups`` selectively **lesions** (disables or removes) units in all groups within the network. The lesioning process is applied to each group based on the given ``lesion_rate``.

The **lesion rate** determines the proportion of units in each group that will be lesioned.

This function is useful for **network degradation studies**, **sparsity control**, or **ablation experiments**.

Arguments
---------

- **lesion_rate** (``float``): The probability (between 0 and 1) that each unit within a group will be lesioned.

  - ``1.0`` means all units will be lesioned.
  - ``0.5`` means approximately half of the units in each group will be lesioned.
  - If the lesion rate is ``0.0``, no lesioning occurs.

Examples
--------

**1. Lesion 30% of units in all groups in the network:**

.. code-block:: python

   net.lesion_all_groups(lesion_rate=0.3)

**2. Completely lesion all groups in the network:**

.. code-block:: python

   net.lesion_all_groups(lesion_rate=1.0)

See Also
--------

- :meth:`lesion_bias_links <your_module.lesion_bias_links>`: Applies a lesion to bias links.
- :meth:`heal_all_groups <your_module.heal_all_groups>`: Restores lesioned units in all groups.
- :meth:`heal_specific_group <your_module.heal_specific_group>`: Restores lesioned units in a specific group.
