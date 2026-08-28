.. _load_example_set:

load_example_set
================

Reads an example file and loads it into an example set.

Usage
-----

.. code-block:: python

   net.load_example_set(
       file_name, 
       name="example",
       proc=False,
       default_input=0,
       active_input=1,
       default_target=0,
       active_target=1,
       num_examples_loaded=None,
       def_s_max_time=example_params.DEF_S_MAXTIME,
       def_s_min_time=example_params.DEF_S_MAXTIME,
       training=True,
       testing=False,
       sort_mode="ORDERED"
   )

Description
-----------

``load_example_set`` loads an example set into the network from a specified file. The function supports both **stored** and **piped** loading modes:

- **Stored Mode (default)**: Loads all examples into memory.

- **Piped Mode** (``proc=True``): Reads examples dynamically from a process or stream.

If the file name ends with ``.gz``, ``.bz``, ``.bz2``, or ``.Z``, it will be automatically decompressed before loading.

By default, the example set name is based on the file name (without the extension or path). However, a specific name can be provided using the ``name`` parameter.

If the network does not have a training set, the loaded set will become the training set. If a training set already exists but no testing set is defined, the new set will automatically be assigned as the testing set.

If ``num_examples_loaded`` is specified, only that many examples will be loaded.

The ``sort_mode`` parameter can be used to control the order in which the examples are processed.

Arguments
---------

- **file_name** (``str``): The name of the file containing the example set. If the file is missing an extension, it will be assumed to be in ``./examples/lens_example_input/``.

- **name** (``str``, optional, default ``"example"``): The name assigned to the example set.

- **proc** (``bool``, optional, default ``False``): If ``True``, examples are **piped** dynamically from an external process rather than loaded all at once.

- **default_input** (``int``): Default value for inactive input units.

- **active_input** (``int``): Value used for active input units.

- **default_target** (``int``): Default value for inactive target units.

- **active_target** (``int``): Value used for active target units.

- **num_examples_loaded** (``int``, optional): If specified, limits the number of examples loaded from the file.

- **def_s_max_time** (``float``, optional): The default maximum time for examples in the set.

- **def_s_min_time** (``float``, optional): The default minimum time for examples in the set.

- **training** (``bool``, optional, default ``True``): If ``True``, the set is added to the **training** sets.

- **testing** (``bool``, optional, default ``False``): If ``True``, the set is added to the **testing** sets.

- **sort_mode** (``str``, optional, default ``"ORDERED"``): Determines how examples are stored and processed:

  - ``"ORDERED"`` → Loads the examples in the original order in which they appear.
  - ``"SHUFFLED"`` → Randomly shuffles the examples upon loading.

Examples
--------

**1. Load an example set from a compressed file and use it for training:**

.. code-block:: python

   net.load_example_set("Examples/digits.ex.gz", name="digits")

**2. Load and append 100 examples from a different file into an existing set:**

.. code-block:: python

   net.load_example_set("set2.ex", name="big_set", num_examples_loaded=100)

**3. Pipe examples dynamically from an external process:**

.. code-block:: python

   net.load_example_set("| generateStuff | formatExamples", name="mySet", proc=True)

See Also
--------