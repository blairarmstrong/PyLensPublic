.. _train:

train
=========

.. the description is stored in the conf.py file

|train|

Usage
-----

.. code-block:: python

   net.train(
       epochs=network_params.PAR_N_numUpdates, 
       batch_size=network_params.PAR_N_batchSize, 
       report_interval=network_params.PAR_N_reportInterval, 
       stop_event=None, 
       parallel_mode=False,
       num_worker=None
   )

Description
-----------

``train`` initializes and executes the **training process** for the network. The training follows these steps:

- **Runs for a specified number of epochs** (``epochs``).
- **Uses mini-batch training with a specified batch size** (``batch_size``).
- **Reports progress at specified intervals** (``report_interval``).
- **Supports both NumPy-based and PyTorch-based optimizers** (with validation checks for learning rate, momentum, and weight decay).
- **Supports parallel training** (``parallel_mode=True``). For details, see :doc:`../parallel`.

During training, network parameters are updated, and performance metrics are reported. If an output path (``net_res_save_path``) is set, results will be saved automatically.

Arguments
---------

- **epochs** (``int``, optional, default ``network_params.PAR_N_numUpdates``): The number of weight updates to perform during training. Must be greater than 0.

- **batch_size** (``int``, optional, default ``network_params.PAR_N_batchSize``): The number of examples processed before a weight update. If set to ``0``, the entire dataset is run before each update.

- **report_interval** (``int``, optional, default ``network_params.PAR_N_reportInterval``): The number of updates between each progress report. Must be greater than ``0``.

- **stop_event** (optional, default ``None``): A threading event that can be used to **interrupt training early**.

- **parallel_mode** (``bool``, optional, default ``False``): If `True`, enables **parallel training**. Uses ``self.num_worker`` workers to distribute computation.

- **num_worker** (optional, default ``None``): Set the number of parallel workers ``self.num_worker`` for parallel training.

Returns
-------

- **total_time** (``float``): The total time taken for training.

Examples
--------

**1. Train the network for 100 epochs with a batch size of 32 and report every 10 updates:**

.. code-block:: python

   net.train(epochs=100, batch_size=32, report_interval=10)

**2. Train with full-batch updates (batch_size=0) and enable parallel training with 3 parallel workers:**

.. code-block:: python

   net.train(epochs=50, batch_size=0, parallel_mode=True, num_worker=3)

**3. Train and allow for external stopping using a threading event:**

.. code-block:: python

   import threading
   stop_event = threading.Event()
   net.train(epochs=200, batch_size=64, stop_event=stop_event)

See Also
--------

- :doc:`use_training_set`: |use_training_set|
- :doc:`set_update_method`: |set_update_method|
- :doc:`update_weights`: |update_weights|
- :doc:`test`: |test|
