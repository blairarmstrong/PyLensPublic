.. _load_network:

load_network
============

Loads a previously saved network from a **pickle** or **JSON** file.

Usage
-----

.. code-block:: python

   net.load_network(fn)

Description
-----------

``load_network`` restores a **previously stored network state** from a given file. It supports loading from both **pickle** and **JSON** formats:

- If the filename ends with **".pickle"**, the function loads a **binary** file using Python's ``pickle`` module.
- If the filename ends with **".json"**, the function loads a **human-readable JSON file**.

After loading, the network metadata is processed using ``from_json()`` to reconstruct the network state.

Arguments
---------

- **fn** (``str``): The filename of the saved network.

Returns
-------

This function does not return any value but updates the network state based on the loaded file.

Examples
--------

**1. Load a network from a pickle file:**

.. code-block:: python

   net.load_network("network.pkl")

**2. Load a network from a JSON file:**

.. code-block:: python

   net.load_network("network.json")

See Also
--------

- :meth:`store_network <your_module.store_network>`: Saves the network to a file.
- :meth:`from_json <your_module.from_json>`: Converts loaded network metadata into a network object.
