.. _lesion_specific_group:

lesion_specific_group
=====================

Applies a lesion to a specific group in the network.

Usage
-----

.. code-block:: python

   net.lesion_specific_group(group_name, lesion_rate=0, lesion_units=None)

Description
-----------

``lesion_specific_group`` selectively **lesions** (disables or removes) units within a specific group in the network. The lesioning process can be controlled in two ways:

- **By probability** (``lesion_rate``): Randomly lesions units based on a given probability.
- **By specific unit indices** (``lesion_units``): Directly selects specific units to lesion.

If both ``lesion_rate`` and ``lesion_units`` are provided, the function will apply both types of lesions.

Arguments
---------

- **group_name** (``str``): The name of the group to be lesioned. Must be a valid existing group in the network.

- **lesion_rate** (``float``, optional, default ``0``): The probability (between 0 and 1) that each unit within the group will be lesioned.

  - If ``1.0``, all units in the group will be lesioned.
  - If ``0.0``, no lesioning occurs.

- **lesion_units** (``List[int]``, optional): A list of unit indices that should be lesioned. If provided, only the specified units will be affected.

Examples
--------

**1. Lesion 30% of units in the "hidden1" group:**

.. code-block:: python

   net.lesion_specific_group("hidden1", lesion_rate=0.3)

**2. Lesion specific units (indices 2, 5, and 8) in the "output" group:**

.. code-block:: python

   net.lesion_specific_group("output", lesion_units=[2, 5, 8])

**3. Fully lesion all units in the "input" group:**

.. code-block:: python

   net.lesion_specific_group("input", lesion_rate=1.0)

See Also
--------

- :meth:`lesion_all_groups <your_module.lesion_all_groups>`: Lesions all groups in the network.
- :meth:`heal_specific_group <your_module.heal_specific_group>`: Restores lesioned units in a specific group.
- :meth:`heal_all_groups <your_module.heal_all_groups>`: Restores lesioned units in all groups.
