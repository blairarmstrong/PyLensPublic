Special Topics: Parallel Training
==========================================

Instead of running your network on a single processing unit, you can split training over multiple processing units (or "workers") that run in parallel. This reduces total training time and makes training large networks more feasible.

PyLens includes built-in support for parallel training, implemented using `Ray Core <https://docs.ray.io/en/latest/ray-core/walkthrough.html>`_.

Using Parallel Training
----------------------------

To enable parallel training, set the ``parallel_mode`` flag to ``True`` in the ``train()`` command and use the ``num_worker`` flag to specify the number of workers. Each worker corresponds to one CPU core:

.. code-block:: python

    xor_net.train(parallel_mode=True, num_worker=3)


Caveats for Parallel Training
---------------------------------

When using the parallel training mode, you cannot directly set other network attributes in the standard Pythonic way. 

For example, instead of:


.. code-block:: python

    xor_net.name = 'good'

you must use the provided `set_properties()` method:

.. code-block:: python

    xor_net.set_properties(name='good')

The `set_properties()` method also supports multiple assignments and assignments to nested objects.

For instance, the following attribute assignments:

.. code-block:: python

    # set network parameters
    digits_net_one.group_criterion_threshold = 0.001
    digits_net_one.test_group_criterion_threshold = 0.001
    digits_net_one.clamp_strength = 1.0
    digits_net_one.init_gain = 0.1
    digits_net_one.final_gain = 1.0
    digits_net_one.anneal_time = 1.0

    # set example set parameters
    digits_net_one.training_sets[0].max_time = 3.0
    digits_net_one.training_sets[0].min_time = 0.0
    digits_net_one.training_sets[0].grace_time = 1.0

can be replaced with a single `set_properties()` call:

.. code-block:: python

    # set network and example parameters
    digits_net_one.set_properties(
        group_criterion_threshold=0.001,
        test_group_criterion_threshold=0.001,
        clamp_strength=1.0,
        init_gain=0.1,
        final_gain=1.0,
        anneal_time=1.0,
        training_sets=[
            {"max_time": 3.0, "min_time": 0.0, "grace_time": 1.0}
        ]
    )


Checking Status of Parallel Workers
-----------------------------------




To check the status of parallel training, open a new terminal, run the following commands:

.. code-block:: shell

    ray status
    ray worker list

``ray status`` outputs the status of node (main server of the network)

``ray actor list`` outputs the status of parallel workers (they are Ray actors). 

For more information of ray command to check status, visit https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html

**Examples**:

**1. Check node status**

Command:

.. code-block:: shell

    ray status

Output:

.. code-block:: shell

    $ ray status

    ======== Autoscaler status: 2026-01-01 01:00:00.000000 ========
    Node status
    ---------------------------------------------------------------
    Active:
     1 node_85b891eb9522ecfed10ecaa1914a5a796bed831e533705e68309ccb0
    Pending:
     (no pending nodes)
    Recent failures:
     (no failures)

    Resources
    ---------------------------------------------------------------
    Usage:
     0.0/8.0 CPU
     0B/6.58GiB memory
     0B/2.00GiB object_store_memory

    Demands:
     (no resource demands)

Here we have one node that is active.

**2. check worker status**

Command:

.. code-block:: shell

    ray actor list

Output:

.. code-block:: shell

    $ ray actor list

    ======== List: 2026-01-01 01:00:00.000000 ========
    Stats:
    ------------------------------
    Total: 3

    Table:
    ------------------------------
        ACTOR_ID                          CLASS_NAME       STATE      JOB_ID  NAME    NODE_ID                                                     PID  RAY_NAMESPACE
     0  01fa2fd6bcd33c1bedf15c2001000000  ParallelNetwork  ALIVE    01000000          85b891eb9522ecfed10ecaa1914a5a796bed831e533705e68309ccb0  85513  1ed16000-8282-4436-bb0
    2-255d5b868836
     1  283bf1bec46d46cb98d2188901000000  ParallelNetwork  ALIVE    01000000          85b891eb9522ecfed10ecaa1914a5a796bed831e533705e68309ccb0  85507  1ed16000-8282-4436-bb0
    2-255d5b868836
     2  d484e6b3f1285ad97a8de57d01000000  ParallelNetwork  ALIVE    01000000          85b891eb9522ecfed10ecaa1914a5a796bed831e533705e68309ccb0  85509  1ed16000-8282-4436-bb0
    2-255d5b868836

Here we have 3 parallel works connected to the same node, each with a separate PID.


