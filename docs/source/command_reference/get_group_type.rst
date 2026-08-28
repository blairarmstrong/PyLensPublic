.. _get_group_type:

get_group_type
==================

.. the description is stored in the conf.py file

|get_group_type|

Usage
-----

.. code-block:: python

   net.get_group_type(
       group_name=None
   )

Description
-----------

``get_group_type`` displays the type, input type, and output type of a group. If no group name is provided, it prints all available base types, input types, and output types.

Groups in PyLens define **collections of units** with specific behaviors. Each group can be assigned a **group type**, determining how it interacts within the network.

Common group types:

- **INPUT** → Receives external data.
- **OUTPUT** → Produces final outputs.
- **HIDDEN** → Internal processing layer.
- **BIAS** → Provides a constant bias input.
- **ELMAN** → Used for Elman recurrent networks.

Arguments
---------

- **group_name** (``str``, optional): Name of the group to query.

Examples
--------

**1. List all available group types, input types, and output types**

.. code-block:: python

   net.get_group_type()

**2. Get the type information for a specific group**

.. code-block:: python

   net.get_group_type("hidden")

See Also
--------

- :doc:`change_group_type`: |change_group_type|