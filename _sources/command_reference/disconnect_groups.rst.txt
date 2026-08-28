.. _disconnect_groups:

disconnect_groups
==============

.. the description is stored in the conf.py file

|disconnect_groups|

Usage
-----

.. code-block:: python

   net.disconnect_groups(
       group1,
       group2,
       link_type=None
   )

Description
-----------

``disconnect_groups`` removes all links of the specified type that project from ``group1`` to ``group2``. If ``link_type`` is ``None``, all links are removed, including Elman projections.

The function returns the number of links removed (as an integer).

Arguments
---------

- **group1** (``str``): Name (or unique identifier) of the group from which links originate.

- **group2** (``str``): Name (or unique identifier) of the group receiving the connections.

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Remove all links from ``input`` to ``hidden``**

.. code-block:: python

   net.disconnect_groups(
       group1="input",
       group2="hidden"
   )

**2. Remove only links of a specific type**

.. code-block:: python

   net.disconnect_groups(
       group1="hidden",
       group2="output",
       link_type="uniform"
   )

See Also
--------

- :doc:`connect_groups`: |connect_groups|
- :doc:`disconnect_group_units`: |disconnect_group_units|
- :doc:`disconnect_units`: |disconnect_units|