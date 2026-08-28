.. _reset_unit_values:

reset_unit_values
===============

.. the description is stored in the conf.py file

|reset_unit_values|

Usage
-----

.. code-block:: python

   group.reset_unit_values(field, value=None)

Description
-----------

``reset_unit_values`` resets a specific **unit-level attribute** (such as inputs, outputs, targets, or derivatives) to a given value.

- If ``value`` is provided, all units in the group are set to this value.
- If ``value`` is ``None``, a **default value** is used:

  - **Target & external input fields** → Default to ``NaN``.
  - **All other fields** → Default to ``0.0``.

Valid fields include:

- ``"input_matrix"`` → Input values.
- ``"output_matrix"`` → Output values.
- ``"input_derivs"`` → Input derivatives.
- ``"output_derivs"`` → Output derivatives.
- ``"target"`` → Target values.
- ``"external_input"`` → External input values.

Arguments
---------

- **field** (``str``): The attribute to reset. Must be one of: ``"input_matrix"``, ``"output_matrix"``, ``"input_derivs"``, ``"output_derivs"``, ``"target"``, or ``"external_input"``.

- **value** (``float``, optional, default ``None``): The value to reset all units to. If ``None``, **default values** are applied based on the field type.

Examples
--------

**1. Reset all outputs in a group to** ``0.0`` **:**

.. code-block:: python

   group.reset_unit_values("output_matrix")

**2. Reset all targets in a group to** ``NaN`` **:**

.. code-block:: python

   group.reset_unit_values("target")

**3. Reset input derivatives to** ``0.0`` **:**

.. code-block:: python

   group.reset_unit_values("input_derivs")

**4. Set all external input values to** ``0.5`` **:**

.. code-block:: python

   group.reset_unit_values("external_input", 0.5)

See Also
--------

- :doc:`copy_unit_values`: |copy_unit_values|
- :doc:`print_unit_values`: |print_unit_values|