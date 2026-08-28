.. _heal_specific_group:

heal_specific_group
===================

Restores lesioned units in a specific group.

Usage
-----

.. code-block:: python

   net.heal_specific_group(
       group_name, 
       heal_all=False, 
       heal_units=None, 
       heal_rate=None
   )

Description
-----------

``heal_specific_group`` restores **lesioned** (disabled) units within a specified group in the network. 

- If *heal_all* is `True`, **all** lesioned units in the group will be fully restored.
- If *heal_units* is provided, **only** the specified unit indices will be healed.
- If *heal_rate* is set, **only a fraction** of lesioned units (based on the proportion) will be healed.

Only one of these parameters (*heal_all*, *heal_units*, or *heal_rate*) should be provided at a time.

Arguments
---------

- **group_name** (``str``):  
  - The name of the group to be healed.
  - Must be a valid existing group in the network.

- **heal_all** (``bool``, optional, default=`False`):  
  - If `True`, **all** lesioned units in the specified group will be restored.

- **heal_units** (``List[int]``, optional):  
  - A list of unit indices in the group that should be healed.
  - If provided, only the specified units will be restored.

- **heal_rate** (``float``, optional):  
  - A proportion (between 0 and 1) of lesioned units to heal in the group.
  - If set, only a fraction of the lesioned units will be restored.

Examples
--------

**1. Fully heal all units in the group "hidden1":**

.. code-block:: python

   net.heal_specific_group("hidden1", heal_all=True)

**2. Heal specific units (index 2 and 5) in the group "hidden2":**

.. code-block:: python

   net.heal_specific_group("hidden2", heal_units=[2, 5])

**3. Heal 30% of lesioned units in the group "output":**

.. code-block:: python

   net.heal_specific_group("output", heal_rate=0.3)

See Also
--------

- :meth:`heal_all_groups <your_module.heal_all_groups>`: Heals all groups in the network.
- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions links in the network.
- :meth:`links_specific_lesion <your_module.links_specific_lesion>`: Lesions specific links between two groups.
