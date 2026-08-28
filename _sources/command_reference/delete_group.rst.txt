.. _delete_group:

delete_group
============

.. the description is stored in the conf.py file

|delete_group|

Usage
-----

.. code-block:: python

   net.delete_group(
       group_names=[],
       delete_all=False
   )

Description
-----------

``delete_group`` deletes a group or a list of groups with the specified name(s) from the network.

Arguments
---------

- **group_names** (``list``, optional): A list of group names to delete.

- **delete_all** (``bool``, optional): If True, deletes all groups except bias.

Examples
--------

**1. Delete specific groups by name**

.. code-block:: python

   net.delete_group(group_names=["hidden1", "hidden2"])

**2. Delete all non-bias groups**

.. code-block:: python

   net.delete_group(delete_all=True)

See Also
--------

- :doc:`add_group`: |add_group|
