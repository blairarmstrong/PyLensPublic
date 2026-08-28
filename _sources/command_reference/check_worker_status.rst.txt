.. _check_worker_status:

check_worker_status
===============

Check the status of parallel workers

Usage
-----

.. code-block:: shell

    ray status
    ray worker list


Description
-----------

To check the status of parallel training, open a new terminal, run the following commands:

``ray status`` outputs the status of node (main server of the network)

``ray actor list`` outputs the status of parallel workers (they are Ray actors). 

For more information of ray command to check status, visit https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html

Examples
--------

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


See Also
--------

- :doc:`Parallel training <../parallel>`: Overview of parallel training in PyLens
