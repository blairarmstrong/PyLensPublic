.. _thaw_group_inputs:

thaw_group_inputs
=================

Unfreezes (thaws) specific or all incoming links to a group.

Usage
-----

.. code-block:: python

   group.thaw_group_inputs(
       incoming_group=None, 
       thaw_all=None, 
       units_indices=None, 
       link_indices=None, 
       bias_indices=None
   )

Description
-----------

``thaw_group_inputs`` **unfreezes incoming links** for a group, allowing weight updates. The function provides several options:

- **Thaw all links** (`thaw_all=True`):  
  - Unfreezes **all links** in the network.

- **Thaw specific units** (`units_indices`):  
  - Unfreezes **all incoming links** for the specified unit indices.

- **Thaw specific connections** (`link_indices`):  
  - Unfreezes **specific links** using `(i, j)` index pairs, where:
    - `i` is the index of a unit in the **previous group**.
    - `j` is the index of a unit in the **current group**.

- **Thaw bias connections** (`bias_indices`):  
  - Unfreezes **bias links** based on specified unit indices.

Arguments
---------

- **incoming_group** (``str``, optional, default=`None`):  
  - The name of the **group sending inputs** to the current group.

- **thaw_all** (``bool``, optional, default=`None`):  
  - If `True`, **all links in the network are unfrozen**.

- **units_indices** (``np.array``, optional, default=`None`):  
  - A **1D NumPy array** specifying units whose incoming links should be unfrozen.

- **link_indices** (``List[Tuple[int, int]]``, optional, default=`None`):  
  - A list of `(i, j)` index pairs specifying which connections to unfreeze.
  - Each pair `(i, j)` refers to a link **from unit `i` in the previous group to unit `j` in the current group**.

- **bias_indices** (``List[int]``, optional, default=`None`):  
  - A list of unit indices for which **bias links should be unfrozen**.

Examples
--------

**1. Unfreeze all links in the network:**

.. code-block:: python

   group.thaw_group_inputs(thaw_all=True)

**2. Unfreeze all incoming links for specific units:**

.. code-block:: python

   group.thaw_group_inputs(incoming_group="hidden1", units_indices=np.array([2, 4, 6]))

**3. Unfreeze specific links between two groups:**

.. code-block:: python

   group.thaw_group_inputs(incoming_group="hidden1", link_indices=[(1, 3), (2, 5)])

**4. Unfreeze bias links for specific units:**

.. code-block:: python

   group.thaw_group_inputs(incoming_group="hidden1", bias_indices=[0, 2, 4])

See Also
--------

- :meth:`freeze <your_module.freeze>`: Freezes a link to prevent weight updates.
- :meth:`unfreeze <your_module.unfreeze>`: Unfreezes a specific link.
- :meth:`thaw_all_links <your_module.thaw_all_links>`: Unfreezes all links in the network.
- :meth:`update_weight <your_module.update_weight>`: Updates the weight of links if they are not frozen.