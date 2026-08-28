.. _test:

test
====

.. the description is stored in the conf.py file

|test|

Usage
-----

.. code-block:: python

    net.test(
        num_examples,
        reset_error=False
    )

Description
-----------

``test`` runs **network evaluation** using the provided testing set. The function:

- **Uses a mini-batch approach** (``num_examples``):  

  - Processes the test set in batches of the specified size.
  - Runs in full-batch mode if num_examples is set to ``0``.

- **Performs model parameter validation**:  

  - Ensures the **learning rate** and **optimizer parameters** are correctly set.  
  - Supports both NumPy-based and PyTorch-based optimizers.

- **Monitors stopping criteria**:  

  - Stops early if the **test error criterion** or **test group criterion** is reached.

- **Computes and reports testing error**:  

  - Calculates the total testing error across all example sets.

Arguments
---------

- **num_examples** (``int``): The number of examples processed before computing an evaluation metric. If set to ``0``, the entire test set is evaluated in a single batch.

- **reset_error** (``bool``, optional, default ``False``): If ``True``, reset the stored error statistics.


Returns
-------

- **total_error_sum** (``float``): The total sum of testing errors across all example sets.

Examples
--------

**1. Evaluate the network using a batch size of 32:**

.. code-block:: python

   net.test(num_examples=32)

**2. Evaluate the network using full-batch mode:**

.. code-block:: python

   net.test(num_examples=0)

See Also
--------

- :doc:`train`: |train|
