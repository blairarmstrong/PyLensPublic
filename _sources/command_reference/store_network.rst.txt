.. _store_network:

store_network
=============

Saves the current network state to a file in either **pickle** or **JSON** format.

Usage
-----

.. code-block:: python

   net.store_network(fn, weight_only=False, format="pickle")

Description
-----------

``store_network`` serializes and saves the **current network state** to a specified file. This allows for **model checkpointing** and later restoration.

- The network metadata is extracted using ``to_json()``.
- **Supports two formats:**

  - ``"pickle"`` (default) → Saves as a **binary** file using Python's ``pickle`` module.
  - ``"json"`` → Saves as a **human-readable JSON file**.

Arguments
---------

- **fn** (``str``): The filename where the network should be saved.

- **weight_only** (``bool``, optional, default ``False``): *(Currently unused in function but can be extended)*. Intended for saving **only network weights** without full metadata.

- **format** (``str``, optional, default ``"pickle"``): The format in which the network should be stored. Options: ``"pickle"``, ``"json"``.

Examples
--------

**1. Save the network as a pickle file:**

.. code-block:: python

   net.store_network("network.pkl")

**2. Save the network as a JSON file:**

.. code-block:: python

   net.store_network("network.json", format="json")

See Also
--------

- :doc:`store_weight`: |store_weight|
