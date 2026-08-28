.. _create_net:

create_net
=================

.. the description is stored in the conf.py file

|create_net|

Usage
-----

.. code-block:: python

   net = simulator.create_net(
       name, 
       time_intervals=1, 
       ticks_per_interval=1, 
       learning_rate=0.1, 
       add_bias=True, 
       type="Standard"
   )

Description
-----------

``create_net`` initializes and returns a **new network instance** from ``simulator`` based on the specified network type. 

.. note::

   ``simulator`` is the top level object in PyLens, which links all the deployed neural networks objects and the associated GUI objects. It is created by ``from src.simulator import Simulator; simulator = Simulator()``

The network type can be one of:

- ``"Standard"`` (default) → Creates a standard feedforward network.
- ``"continuous"`` → Creates a **ContinuousNetwork**.
- ``"srbptt"`` → Creates an **SRBPTTNetwork**.
- ``"boltzmann"`` → Creates a **Boltzmann Machine**.

If an invalid type is provided, an **error is raised**.

Arguments
---------

- **name** (``str``): The name of the new network.

- **time_intervals** (``int``, optional, default ``1``): The number of **time intervals** for recurrent networks.

- **ticks_per_interval** (``int``, optional, default ``1``): The number of **ticks per time interval**.

- **learning_rate** (``float``, optional, default ``0.1``): The **learning rate** for the network.

- **add_bias** (``bool``, optional, default ``True``): Whether to **include a bias unit** in the network.

- **type** (``str``, optional, default ``"Standard"``): The type of network to create. Options are: "Standard", "continuous", "srbptt", "boltzmann".

Returns
-------

- **net** (``Network`` or subclass): A newly created network instance of the specified type.

Examples
--------

**1. Create a standard network:**

.. code-block:: python

   simulator = Simulator()
   net = simulator.create_net("my_network")

**2. Create a continuous network with custom time intervals:**

.. code-block:: python

   simulator = Simulator()
   net = simulator.create_net("continuous_net", time_intervals=10, type="continuous")

See Also
--------

- :doc:`add_net`: |add_net|
- :doc:`use_net`: |use_net|
- :doc:`delete_net`: |delete_net|
