.. _order_groups:

order_groups
==================

.. the description is stored in the conf.py file

|order_groups|

Usage
-----

.. code-block:: python

   net.order_groups(
       group_order
   )

Description
-----------

``order_groups`` sets the order of the groups in the network's group array and thus the order in which the groups are updated. This is crucial for standard networks but less important for continuous networks because continuous networks are updated synchronously.

All groups in the network must be included in ``group_order``. If any group is missing, the method raises an exception.

For simple recurrent networks, it is important that context groups come before their source group.

Arguments
---------

- **group_order** (``List[str]``): The desired order of the groups in the network.

Examples
--------

**1. Set a full update order explicitly**

.. code-block:: python

   net.order_groups(["bias", "input", "hidden", "output"])

See Also
--------

- :doc:`add_group`: |add_group|
- :doc:`delete_group`: |delete_group|