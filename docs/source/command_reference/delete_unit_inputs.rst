.. _delete_unit_inputs:

delete_unit_inputs
======================

.. the description is stored in the conf.py file

|delete_unit_inputs|

Usage
-----

.. code-block:: python

   net.delete_unit_inputs(
       incoming_group,
       incoming_unit_idx,
       link_type=None
   )


Description
-----------

``delete_unit_inputs`` removes all incoming links to the specified units of a group. If ``link_type`` is provided, only inputs of that type are removed.

The function returns the number of input connections disabled (as an integer).

Arguments
---------

- **incoming_group** (``str``): Name (or unique identifier) of the group receiving the connections.

- **incoming_unit_idx** (``list``): List of indices of the units in the incoming group that receive the connections.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Remove all incoming connections to selected units**

.. code-block:: python

   net.delete_unit_inputs(
       incoming_group="hidden",
       incoming_unit_idx=[1, 3]
   )

**2. Remove selected unit inputs for one link type**

.. code-block:: python

   net.delete_unit_inputs(
       incoming_group="output",
       incoming_unit_idx=[0],
       link_type="uniform"
   )

See Also
--------

- :doc:`delete_links`: |delete_links|
- :doc:`delete_group_inputs`: |delete_group_inputs|
- :doc:`delete_group_outputs`: |delete_group_outputs|
- :doc:`delete_unit_outputs`: |delete_unit_outputs|
