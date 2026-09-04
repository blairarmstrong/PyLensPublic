.. _connect_group_to_unit:

connect_group_to_unit
=====================

.. the description is stored in the conf.py file

|connect_group_to_unit|

Usage
-----

.. code-block:: python

   net.connect_group_to_unit(
       outgoing_group,
       incoming_group,
       target_unit_idx,
       initialization="uniform",
       rand_mean=0.0,
       rand_range=None,
       proj_type="full",
       link_type=None,
       lesion_rate=None,
       dropout_rate=None,
       perma_lesion_rate=None,
       bidirectional=False
   )

Description
-----------

``connect_group_to_unit`` connects all units in the **outgoing group** to the specified unit(s) in the **incoming group**. This is implemented by calling ``connect_units(...)`` with all outgoing indices.

By default, newly created links have a type designated by ``link_type``.

The ``rand_mean`` and ``rand_range`` parameters control how new link weights are initialized. Typically, the weights will be chosen randomly from a uniform distribution with center (``rand_mean``) and half-width (``rand_range``). If these parameters are omitted, the network default randomization settings may be used.

For details on connectivity patterns (the ``proj_type`` parameter), see :doc:`connect_groups`.

Set ``bidirectional=True`` to automatically form a reverse projection from the units in the ``incoming_group`` to the ``outgoing_group`` (with the same ``link_type`` and randomization parameters).

Arguments
---------

- **outgoing_group** (``str``): Name (or unique identifier) of the group from which links will originate.

- **incoming_group** (``str``): Name (or unique identifier) of the group receiving the connections.

- **target_unit_idx** (``list``): List of indices of the units in the incoming group that receive the connections.

- **initialization** (``str``, optional, default ``uniform``): Random initialization of the weights between groups.

- **rand_mean** (``float``, optional, default ``0``): Mean (center) of the distribution used to initialize newly created weights.

- **rand_range** (``float``, optional, default ``None``): Half-width (range) of the distribution for the link weights, i.e., weights are initialized in the interval ``[rand_mean - rand_range, rand_mean + rand_range]``. If ``None``, the network default is used.

- **proj_type** (``str``, optional, default ``"full"``): Specifies the connectivity pattern. Common values are: ``"full"``, ``"random"``, ``"fixed_in"``, ``"fixed_out"``, ``"fair"``, ``"fan"``, ``"one_to_one"``, etc.

- **link_type** (``str``, optional, default ``None``): Type or label assigned to the created links.

- **lesion_rate** (``float``, optional, default ``None``): Rate at which connections or units are lesioned (disabled). If ``None``, no lesioning is applied at connection time.

- **dropout_rate** (``float``, optional, default ``None``): Probability of dropping (ignoring) the output of a connection. If ``None``, no dropout is applied.

- **perma_lesion_rate** (``float``, optional, default ``None``): Similar to *lesion_rate* but might indicate permanent or persistent lesioning. If ``None``, no permanent lesions are applied.

- **bidirectional** (``bool``, optional, default ``False``): If ``True``, whenever a forward link is formed, a reverse link is also formed from the ``incoming_group`` to the ``outgoing_group``.

Examples
--------

**1. Connect all hidden units to a single output unit**

.. code-block:: python

   net.connect_group_to_unit(
       outgoing_group="hidden",
       incoming_group="output",
       target_unit_idx=[0]
   )

**2. Connect to multiple target units with lesion/dropout settings**

.. code-block:: python

   net.connect_group_to_unit(
       outgoing_group="input",
       incoming_group="hidden",
       target_unit_idx=[0, 1, 2],
       initialization="uniform",
       lesion_rate=0.1,
       dropout_rate=0.2
   )

See Also
--------

- :doc:`connect_groups`: |connect_groups|
- :doc:`connect_units`: |connect_units|
