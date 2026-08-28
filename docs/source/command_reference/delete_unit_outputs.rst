.. _delete_unit_outputs:

delete_unit_outputs
====================

.. the description is stored in the conf.py file

|delete_unit_outputs|

Usage
-----

.. code-block:: python

   net.delete_unit_outputs(
       outgoing_group,
       outgoing_unit_idx,
       link_type=None
   )


Description
-----------

``delete_unit_outputs`` removes all outgoing links from the specified units of a group. If ``link_type`` is provided, only outputs of that type are removed.

The function returns the number of output connections disabled (as an integer).

Arguments
---------

- **outgoing_group** (``str``): Name (or unique identifier) of the group from which the links originate.

- **outgoing_unit_idx** (``list``): List of indices of the units in the outgoing group from which the links originate.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Remove all outgoing connections from selected source units**

.. code-block:: python

   net.delete_unit_outputs(
       outgoing_group="hidden",
       outgoing_unit_idx=[0, 2]
   )

**2. Remove selected unit outputs for one link type**

.. code-block:: python

   net.delete_unit_outputs(
       outgoing_group="input",
       outgoing_unit_idx=[1],
       link_type="uniform"
   )

See Also
--------

- :doc:`delete_links`: |delete_links|
- :doc:`delete_group_inputs`: |delete_group_inputs|
- :doc:`delete_group_outputs`: |delete_group_outputs|
- :doc:`delete_unit_inputs`: |delete_unit_inputs|
