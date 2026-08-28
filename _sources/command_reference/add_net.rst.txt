.. _add_net:

add_net
=============

.. the description is stored in the conf.py file

|add_net|

Usage
-----

.. code-block:: python

   simulator.add_net(net)

Description
-----------

``add_net`` registers a **network instance** to the simulator. This function:

- **Ensures the network is only added once** to prevent duplication.
- **Assigns the simulator to the network** (``net.simulator = self``).
- **Sets the active network** if none has been assigned.

This function is useful for managing **multiple networks** within a simulation environment.

Arguments
---------

- **net** (``Network``): The network instance to be added.

Examples
--------

**1. Add a network to the simulator:**

.. code-block:: python

   net = simulator.create_net("my_network")
   simulator.add_net(net)

**2. Ensure the first added network becomes the active network:**

.. code-block:: python

   net1 = simulator.create_net("network1")
   simulator.add_net(net1)  # net1 is now the active network
   net2 = simulator.create_net("network2")
   simulator.add_net(net2)  # net2 is added, but net1 remains active

See Also
--------

- :doc:`create_net`: |create_net|
- :doc:`use_net`: |use_net|
- :doc:`delete_net`: |delete_net|
