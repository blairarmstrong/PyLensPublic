.. _print_unit_values:

print_unit_values
=================

.. the description is stored in the conf.py file

|print_unit_values|

Usage
-----

.. code-block:: python

   net.print_unit_values(
       filename,
       group_names="*",
       append=False
   )

Description
-----------

``print_unit_values`` prints information about each unit in a given list of groups to a file. Lesioned units are ignored when the values are printed.

If a file does not already exist, it is created. Supported file types are .gz and .bz2.

The information printed includes: ``input_matrix``, ``output_matrix``, ``target``, ``input_derivs``, ``output_derivs``, and ``external_input``.

Unit values are printed in the following format: ``group_name unit_index field_name value(s)``.

The command returns the number of lines written.

Arguments
---------

- **filename** (``str``): Name of the file that the information is written to. Supports .gz and .bz2.

- **group_names** (``str`` or ``list``, optional, default ``"*"``): Groups whose unit values are printed. Use ``"*"`` (or omit) to include all groups.

- **append** (``bool``): If ``False``, overwrites the existing file. If ``True``, append to the existing file.

Examples
--------

**1. Write values for all groups to a new file**

.. code-block:: python

   net.print_unit_values("unit_values.txt")

**2. Append values for selected groups to a compressed file**

.. code-block:: python

   net.print_unit_values(
       "unit_values.bz2",
       group_names=["hidden", "output"],
       append=True
   )

See Also
--------

- :doc:`copy_unit_values`: |copy_unit_values|
- :doc:`reset_unit_values`: |reset_unit_values|