.. _heal_all_links:

heal_all_links
==============

Restores lesioned links across all groups in the network.

Usage
-----

.. code-block:: python

   net.heal_all_links(heal_rate=None)

Description
-----------

``heal_all_links`` restores **lesioned links** in all groups within the network. The healing process can be controlled in two ways:

- **Full Healing**: If no rate is specified (``heal_rate=None``), all lesioned links in the network will be fully restored.

- **Partial Healing**: A proportion of links will be healed based on the given ``heal_rate``.
  
This function is useful for **gradually restoring connectivity** in a network after lesioning.

Arguments
---------

- **heal_rate** (``float``, optional, default ``None``): The proportion (between 0 and 1) of lesioned links to heal.

  - If `None`, **all** lesioned links are fully restored.
  - If `0.5`, restores approximately 50% of the lesioned links.

Examples
--------

**1. Fully heal all links in the network:**

.. code-block:: python

   net.heal_all_links()

**2. Heal 50% of lesioned links across all groups:**

.. code-block:: python

   net.heal_all_links(heal_rate=0.5)

See Also
--------

- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions all links in the network.
- :meth:`heal_specific_group <your_module.heal_specific_group>`: Restores lesioned units in a specific group.
- :meth:`heal_all_groups <your_module.heal_all_groups>`: Restores lesioned units in all groups.
