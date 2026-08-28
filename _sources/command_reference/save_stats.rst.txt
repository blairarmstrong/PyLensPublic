.. _save_stats:

save_stats
=============

.. the description is stored in the conf.py file

|save_stats|

Usage
-----

.. code-block:: python

   net.save_stats(file_path="training_stats.csv")

Description
-----------

``save_stats`` saves the network's **training statistics** to a specified file. This function:

- Calls ``stats_plotter.save_stats(file_path)`` to handle the saving process.
- Exports **performance metrics** such as loss, accuracy, and other recorded statistics.

By default, the file is saved as `"training_stats.csv"` in the current directory.

Arguments
---------

- **file_path** (``str``, optional, default ``"training_stats.csv"``): The path where the training statistics should be saved.

Examples
--------

**1. Save training statistics to the default file:**

.. code-block:: python

   net.save_stats()

**2. Save training statistics to a custom file:**

.. code-block:: python

   net.save_stats("results/network_training.csv")

See Also
--------

- :doc:`train`: |train|
- :doc:`test`: |test|
