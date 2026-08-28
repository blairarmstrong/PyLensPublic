.. _delete_group_outputs:

delete_group_outputs
=======================

.. the description is stored in the conf.py file

|delete_group_outputs|

Usage
-----

.. code-block:: python

   net.delete_group_outputs(
       outgoing_group,
       link_type=None
   )


Description
-----------

``delete_group_outputs`` removes all outgoing links from every unit of a group. If ``link_type`` is provided, only outputs of that type are removed.

This is the group-wide version of ``delete_unit_outputs``.

The function returns the number of output connections disabled (as an integer).

Arguments
---------

- **outgoing_group** (``str``): Name (or unique identifier) of the group from which the links originate.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Remove all outgoing connections from a group**

.. code-block:: python

   net.delete_group_outputs(
       outgoing_group="hidden"
   )

**2. Remove only outgoing connections of one link type**

.. code-block:: python

   net.delete_group_outputs(
       outgoing_group="input",
       link_type="uniform"
   )

See Also
--------

- :doc:`delete_links`: |delete_links|
- :doc:`delete_group_inputs`: |delete_group_inputs|
- :doc:`delete_unit_inputs`: |delete_unit_inputs|
- :doc:`delete_unit_outputs`: |delete_unit_outputs|
