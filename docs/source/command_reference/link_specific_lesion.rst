.. _link_specific_lesion:

link_specific_lesion
====================

Applies a lesion to specific links between two groups in the network.

Usage
-----

.. code-block:: python

   net.link_specific_lesion(
       outgoing_group, 
       incoming_group, 
       proj_type, 
       cartesian=False, 
       lesion_rate=0, 
       links_to_lesion=None
   )

Description
-----------

``link_specific_lesion`` selectively **lesions** (disables or removes) links between two specified groups in the network. This allows for fine-grained control over which connections are affected.

Lesioning can be done in two ways:

- **By probability** (``lesion_rate``): Randomly lesions links based on a given probability.
- **By specific indices** (``links_to_lesion``): Directly selects specific links to lesion.

If both ``lesion_rate`` and ``links_to_lesion`` are provided, the function will apply both types of lesions.

Arguments
---------

- **outgoing_group** (``str``): The name of the group that sends data forward.

- **incoming_group** (``str``): The name of the group receiving data.

- **proj_type** (``str``): The type of projection between the two groups (e.g., "full", "random", "one_to_one"). Used to determine how lesioning is applied.

- **cartesian** (``bool``, optional, default ``False``): If ``True``, indices for ``links_to_lesion`` are treated as Cartesian coordinates.

- **lesion_rate** (``float``, optional, default ``0``): The probability (between 0 and 1) that each link will be lesioned. If set, lesions links probabilistically.

- **links_to_lesion** (``List[int]``, optional): A list of specific link indices to be lesioned. If provided, only these links will be affected.

Examples
--------

**1. Lesion 30% of links between "hidden1" and "output":**

.. code-block:: python

   net.link_specific_lesion("hidden1", "output", proj_type="full", lesion_rate=0.3)

**2. Lesion specific links (indices 2, 5, and 8) between "input" and "hidden":**

.. code-block:: python

   net.link_specific_lesion("input", "hidden", proj_type="random", links_to_lesion=[2, 5, 8])

**3. Lesion using Cartesian indexing for a structured projection:**

.. code-block:: python

   net.link_specific_lesion("input", "hidden", proj_type="grid", cartesian=True, links_to_lesion=[(0, 1), (2, 3)])

See Also
--------

- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions all links in the network.
- :meth:`heal_specific_group <your_module.heal_specific_group>`: Restores lesioned units in a specific group.
- :meth:`remove_links_specific_noise <your_module.remove_links_specific_noise>`: Removes noise from links between two groups.
