.. _copy_connect:

copy_connect
=============

.. the description is stored in the conf.py file

|copy_connect|

Usage
-----

.. code-block:: python

   net.copy_connect(
       source_group,
       copy_group,
       field
   )

Description
-----------

``copy_connect`` wires a COPY transform on ``copy_group`` so it reads a specific ``field`` from ``source_group``.

The method looks for the first empty COPY slot on the destination group in this order:

- an ``In_Copy`` input transform
- otherwise an ``Out_Copy`` output transform

It then sets:

- ``transform.source_group``
- ``transform.source_field``

The source and copy groups must have the same number of units.

Valid source fields are: ``inputs``, ``externalInputs``, ``outputs``, ``targets``, ``inputDerivs``, and ``outputDerivs``.

Arguments
---------

- **source_group** (``str`` or ``Group``): The group from which values are copied.

- **copy_group** (``str`` or ``Group``): The destination group that contains an empty COPY transform slot.

- **field** (``str``): The source group's field to copy from. Valid options are: ``inputs``, ``externalInputs``, ``outputs``, ``targets``, ``inputDerivs``, and ``outputDerivs``.

Examples
--------

**1. Copy outputs from ``hidden`` into an ``In_Copy`` transform on ``context``**

.. code-block:: python

   net.copy_connect(
       source_group="hidden",
       copy_group="context",
       field="outputs"
   )

**2. Copy targets from ``teacher`` into an ``Out_Copy`` transform on ``student``**

.. code-block:: python

   net.copy_connect(
       source_group="teacher",
       copy_group="student",
       field="targets"
   )

See Also
--------

- :doc:`elman_connect`: |elman_connect|