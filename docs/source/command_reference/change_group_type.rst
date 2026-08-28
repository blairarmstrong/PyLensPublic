.. _change_group_type:

change_group_type
==================

.. the description is stored in the conf.py file

|change_group_type|

Usage
-----

.. code-block:: python

   net.change_group_type(
       group_name,
       new_input_transforms=[],
       new_output_transforms=[]
   )

Description
-----------

``change_group_type`` can be used to add new input and/or output transforms to a group. Note that the new transforms are **appended** to the group's existing transform lists rather than replacing them.

Arguments
---------

- **group_name** (``str``): The name of the group to change.

- **new_input_transforms** (``list``, optional): A list of input transform names to append. Valid names include: ``"dot"``, ``"product"``, ``"soft_clamp"``, ``"in_copy"``, ``"in_integr"``, ``"distance"``, ``"boltzmann"``.

- **new_output_transforms** (``list``, optional): A list of output transform names to append. Valid names include: ``"sigmoid"``, ``"linear"``, ``"tanh"``, ``"soft_max"``, ``"noise"``, ``"hard_clamp"``, ``"cropped"``, ``"gaussian"``, ``"exponential"``, ``"elman_clamp"``, etc.

Examples
--------

**1. Append a noise output transform to the hidden group**

.. code-block:: python

   net.change_group_type(
       "hidden",
       new_output_transforms=["noise"]
   )

**2. Append input and output transforms at the same time**

.. code-block:: python

   net.change_group_type(
       "hidden",
       new_input_transforms=["soft_clamp"],
       new_output_transforms=["linear"]
   )

See Also
--------

- :doc:`get_group_type`: |get_group_type|