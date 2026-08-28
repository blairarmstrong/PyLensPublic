.. _disconnect_units:

disconnect_units
==============

.. the description is stored in the conf.py file

|disconnect_units|

Usage
-----

.. code-block:: python

   net.disconnect_units(
       outgoing_group,
       outgoing_unit_idx,
       incoming_group,
       incoming_unit_idx,
       link_type=None
   )

Description
-----------

``disconnect_units`` between two units by clearing the corresponding entries in the connection mask. If ``link_type`` is ``None``, the function is applied to all link types. If a link’s mask becomes entirely zero after modification, the link is removed.

This function returns the number of unit-pairs disconnected (as an integer).

Arguments
---------

- **outgoing_group** (``str``): Name (or unique identifier) of the group from which the links originate.

- **outgoing_unit_idx** (``list``): List of indices of the units in the outgoing group from which the links originate.

- **incoming_group** (``str``): Name (or unique identifier) of the group receiving the connections.

- **incoming_unit_idx** (``list``): List of indices of the units in the incoming group that receive the connections.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Disconnect one specific pair of units**

.. code-block:: python

   net.disconnect_units(
       outgoing_group="input",
       outgoing_unit_idx=[0],
       incoming_group="hidden",
       incoming_unit_idx=[2]
   )

**2. Disconnect multiple unit pairs for one link type**

.. code-block:: python

   net.disconnect_units(
       outgoing_group="hidden",
       outgoing_unit_idx=[0, 1],
       incoming_group="output",
       incoming_unit_idx=[0, 1],
       link_type="uniform"
   )

See Also
--------

- :doc:`connect_units`: |connect_units|
- :doc:`disconnect_groups`: |disconnect_groups|
- :doc:`disconnect_group_units`: |disconnect_group_units|