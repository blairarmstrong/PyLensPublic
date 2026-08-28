.. _disconnect_group_units:

disconnect_group_units
==============

.. the description is stored in the conf.py file

|disconnect_group_units|

Usage
-----

.. code-block:: python

   net.disconnect_group_units(
       outgoing_group,
       incoming_group,
       incoming_unit_idx,
       link_type=None
   )

Description
-----------

``disconnect_group_units`` disconnects links from all units in ``outgoing_group`` to selected unit indices in ``incoming_group``. If ``link_type`` is ``None``, it applies to all link types.

The function returns the number of unit-pairs disconnected (as an integer).

Arguments
---------

- **outgoing_group** (``str``): Name (or unique identifier) of the group from which the links originate.

- **incoming_group** (``str``): Name (or unique identifier) of the group receiving the connections.

- **incoming_unit_idx** (``list``): List of indices of the units in the incoming group that receive the connections.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Disconnect all input-to-hidden connections into hidden unit 3**

.. code-block:: python

   net.disconnect_group_units(
       outgoing_group="input",
       incoming_group="hidden",
       incoming_unit_idx=[3]
   )

**2. Disconnect selected output units for one link type**

.. code-block:: python

   net.disconnect_group_units(
       outgoing_group="hidden",
       incoming_group="output",
       incoming_unit_idx=[0, 1],
       link_type="uniform"
   )

See Also
--------

- :doc:`connect_group_to_unit`: |connect_group_to_unit|
- :doc:`disconnect_groups`: |disconnect_groups|
- :doc:`disconnect_units`: |disconnect_units|