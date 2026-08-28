.. _heal_all_groups:

heal_all_groups
===============

Restores lesioned units in all groups within the network.

Usage
-----

.. code-block:: python

   net.heal_all_groups(p=None)

Description
-----------

``heal_all_groups`` restores **lesioned** (disabled) units across all groups in the network. 

- If no proportion (*p*) is specified, **all** lesioned units in every group are fully restored.
- If a proportion (*p*) is given, only that fraction of lesioned units will be healed within each group.

This function is useful for gradually reintroducing units into the network after a lesioning process.

Arguments
---------

- **p** (``float``, optional):  
  - The proportion (between 0 and 1) of lesioned units to heal in each group.  
  - If ``None`` (default), **all** lesioned units will be restored.

Examples
--------

**1. Fully heal all groups in the network:**

.. code-block:: python

   net.heal_all_groups()

**2. Heal 50% of lesioned units in each group:**

.. code-block:: python

   net.heal_all_groups(p=0.5)

See Also
--------

- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions links in the network.
- :meth:`links_specific_lesion <your_module.links_specific_lesion>`: Lesions specific links between two groups.
