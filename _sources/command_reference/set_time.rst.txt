.. _set_time:

set_time
=====================

.. the description is stored in the conf.py file

|set_time|

Usage
-----

.. code-block:: python

    sim.set_time(
        ticks_per_interval=None, time_intervals=None
    )

Description
-----------

With no arguments this returns the values of the active network's ``time_intervals`` and ``ticks_per_interval`` parameters.

Otherwise, this changes one or more of these values and returns the new values. All values are integers. 

Arguments
---------

- **time_intervals** (``int``): The time interval of the network.

- **ticks_per_interval** (``int``): The ticks per interval of the network.
