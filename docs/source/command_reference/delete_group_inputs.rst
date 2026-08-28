.. _delete_group_inputs:

delete_group_inputs
======================

.. the description is stored in the conf.py file

|delete_group_inputs|

Usage
-----

.. code-block:: python

   net.delete_group_inputs(
       incoming_group,
       link_type=None
   )


Description
-----------

``delete_group_inputs`` removes all incoming links to every unit of a group. If ``link_type`` is provided, only inputs of that type are removed.

This is the group-wide version of ``delete_unit_inputs``.

The function returns the number of input connections disabled (as an integer).

Arguments
---------

- **incoming_group** (``str``): Name (or unique identifier) of the group receiving the connections.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Remove all incoming connections to a group**

.. code-block:: python

   net.delete_group_inputs(
       incoming_group="hidden"
   )

**2. Remove only incoming connections of one link type**

.. code-block:: python

   net.delete_group_inputs(
       incoming_group="output",
       link_type="uniform"
   )

See Also
--------

- :doc:`delete_links`: |delete_links|
- :doc:`delete_group_outputs`: |delete_group_outputs|
- :doc:`delete_unit_inputs`: |delete_unit_inputs|
- :doc:`delete_unit_outputs`: |delete_unit_outputs|
