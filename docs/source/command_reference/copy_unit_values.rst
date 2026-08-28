.. _copy_unit_values:

copy_unit_values
=================

.. the description is stored in the conf.py file

|copy_unit_values|

Usage
-----

.. code-block:: python

   net.copy_unit_values(
       src_group_name,
       dst_group_name=None,
       src_field="outputs",
       dst_field=None,
       update_cache=True,
       require_same_units=True
   )

Description
-----------

``copy_unit_values`` copies unit values from one group's field to another field in the same or a different group. It could be used to copy the inputs of a group to its outputs or the outputs of one group to the targets of another.

The copy is instantaneous; this does not set up a permanent connection.

.. note::

   The parameters ``src_field``, ``dst_field``, ``update_cache``, and ``require_same_units`` are **keyword-only** arguments (they follow a bare ``*`` in the signature). They must be passed by name, not by position.

Arguments
---------

- **src_group_name** (``str``): Name of the source group.

- **dst_group_name** (``str``, optional): Name of the destination group. If ``None``, defaults to ``src_group_name``.

- **src_field** (``str``, keyword-only, default ``"outputs"``): Source field name. One of: ``"inputs"``, ``"outputs"``, ``"targets"``, ``"input_derivs"``, ``"output_derivs"``, ``"external_input"``.

- **dst_field** (``str``, keyword-only, optional): Destination field. Defaults to ``src_field``.

- **update_cache** (``bool``, keyword-only, default ``True``): If ``True`` and writing to outputs, refresh cached outputs.

- **require_same_units** (``bool``, keyword-only, default ``True``): If ``True``, require that both groups have the same number of units.

Examples
--------

**1. Copy outputs from one group to another group's targets**

.. code-block:: python

   net.copy_unit_values(
       "hidden",
       "output",
       src_field="outputs",
       dst_field="targets"
   )

**2. Copy outputs to inputs within the same group**

.. code-block:: python

   net.copy_unit_values(
       "hidden",
       src_field="outputs",
       dst_field="inputs"
   )

See Also
--------

- :doc:`print_unit_values`: |print_unit_values|
- :doc:`reset_unit_values`: |reset_unit_values|