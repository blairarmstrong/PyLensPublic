.. _delete_net:

delete_net
==========

.. the description is stored in the conf.py file

|delete_net|

Usage
-----

.. code-block:: python

   simulator.delete_net(net)

Description
-----------

``delete_net`` removes a **network instance** from the simulator's list of managed networks. If the deleted network is the **active network** and at least one network remains, the first remaining network is set as active.

- **Removes the specified network (``net``)** from the simulator.
- **Updates the active network** if the deleted network was previously active.

Arguments
---------

- **net** (``Network``): The network instance to be removed.

Examples
--------

**1. Delete a network from the simulator:**

.. code-block:: python

   net = simulator.create_net("my_network")
   simulator.add_net(net)
   simulator.delete_net(net)

**2. Ensure the first remaining network becomes active if the active network is deleted:**

.. code-block:: python

   sim = Simulator()
   net1 = sim.create_net("network1")
   net2 = sim.create_net("network2")
   sim.add_net(net1)
   sim.add_net(net2)

   sim.delete_net(net1)  # net2 now becomes the active network

See Also
--------

- :doc:`add_net`: |add_net|
- :doc:`delete_group`: |delete_group|
