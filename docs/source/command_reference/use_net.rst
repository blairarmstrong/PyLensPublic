.. _use_net:

use_net
=============

.. the description is stored in the conf.py file

|use_net|

Usage
-----

.. code-block:: python

   simulator.use_net(net_name)

Description
-----------

``use_net`` switches the simulator to use the specified network as the **active network**.

If the given network name exists within the simulator’s network list, it is set as active.  
Otherwise, the function returns ``False``.

Arguments
---------

- **net_name** (``str``): The **name** of the network to activate.

Returns
-------

- ``True`` → If the network was successfully set as the active network.
- ``False`` → If the network was **not found** in the simulator.

Examples
--------

**1. Set "my_network" as the active network:**

.. code-block:: python

   success = simulator.use_net("my_network")
   if success:
       print("Switched to my_network")
   else:
       print("Network not found")

**2. Ensure a network is activated before performing operations:**

.. code-block:: python

   if simulator.use_net("experiment_net"):
       simulator.train()
   else:
       print("Network not found. Please check the name.")

See Also
--------

- :doc:`create_net`: |create_net|
- :doc:`add_net`: |add_net|
- :doc:`delete_net`: |delete_net|
