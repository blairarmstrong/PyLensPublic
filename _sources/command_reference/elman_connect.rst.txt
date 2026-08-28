.. _elman_connect:

elman_connect
=============

.. the description is stored in the conf.py file

|elman_connect|

Usage
-----

.. code-block:: python

   net.elman_connect(source_group, context_group)

Description
-----------

``elman_connect`` wires an Elman source group into an existing **ELMAN context group** by setting the source used by the context group's ``Elman_Clamp`` output transform.

- It does **not** create a new link object.
- It requires the context group to already have an ``Elman_Clamp`` transform.
- It validates that the source group has at least as many units as the context group.

This command returns ``True`` on success.

Arguments
---------

- **source_group** (``str`` or ``Group``): The group whose outputs are used as Elman context.

- **context_group** (``str`` or ``Group``): The Elman context group that contains an ``Elman_Clamp`` transform.

Examples
--------

**1. Connect hidden1 as the Elman source for hidden2:**

.. code-block:: python

   net.elman_connect("hidden1", "hidden2")

**2. Connect using group objects instead of names:**

.. code-block:: python

   src = net.get_group_by_name("hidden1")
   ctx = net.get_group_by_name("hidden2")
   net.elman_connect(src, ctx)

See Also
--------

- :doc:`connect_groups`: |connect_groups|