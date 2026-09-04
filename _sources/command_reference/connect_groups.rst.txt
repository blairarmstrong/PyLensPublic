.. _connect_groups:

connect_groups
==============

.. the description is stored in the conf.py file

|connect_groups|

Usage
-----

.. code-block:: python

   net.connect_groups(
       outgoing_group,
       incoming_group,
       initialization="uniform",
       rand_mean=0,
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

``connect_groups`` creates sets of links between two groups, with an option to make the connection bidirectional. By default, a ``"full"`` projection connects every unit in the ``outgoing_group`` to every unit in the ``incoming_group``.

The ``proj_type`` parameter controls connectivity pattern. Current supported values are:

- **full**: Dense projection from every sending unit to every receiving unit.
- **random**: Random sparse projection (implementation-defined density).
- **one-to-one**: Connect corresponding units pairwise.
- **elman**: Connect a source to elman conext group through ``elman_connect()``.

The ``rand_mean`` and ``rand_range`` parameters control how new link weights are initialized. Typically, the weights will be chosen randomly from a uniform distribution with center (``rand_mean``) and half-width (``rand_range``). If these parameters are omitted, the network default randomization settings may be used.

By default, newly created links have a type designated by ``link_type``. 

Set ``bidirectional=True`` to automatically form a reverse projection from ``incoming_group`` to ``outgoing_group`` (with the same ``link_type`` and randomization parameters).

Arguments
---------

- **outgoing_group** (``str``): Name (or unique identifier) of the group from which the links will originate.

- **incoming_group** (``str``): Name (or unique identifier) of the group receiving the connections.

- **initialization** (``str``, optional, default ``uniform``): Random initialization of the weights between groups.

- **rand_mean** (``float``, optional, default ``0``): Mean (center) of the distribution used to initialize newly created weights.

- **rand_range** (``float``, optional, default ``None``): Half-width (range) of the distribution for the link weights, i.e., weights are initialized in the interval ``[rand_mean - rand_range, rand_mean + rand_range]``. If ``None``, the network default is used.

- **proj_type** (``str``, optional, default ``"full"``): Connectivity pattern. Supported values are ``"full"``, ``"random"``, and ``"one-to-one"``.

- **link_type** (``str``, optional): Type or label assigned to the created links.

- **lesion_rate** (``float``, optional, default ``None``): Rate at which connections or units are lesioned (disabled). If ``None``, no lesioning is applied at connection time.

- **dropout_rate** (``float``, optional, default ``None``): Probability of dropping (ignoring) the output of a connection. If ``None``, no dropout is applied.

- **perma_lesion_rate** (``float``, optional, default ``None``): Similar to *lesion_rate* but might indicate permanent or persistent lesioning. If ``None``, no permanent lesions are applied.

- **bidirectional** (``bool``, optional, default ``False``): If ``True``, whenever a forward link is formed, a reverse link is also formed from the ``incoming_group`` to the ``outgoing_group``.

Examples
--------

**1. Create a full (dense) projection from group "input" to group "hidden"**:

.. code-block:: python

   net.connect_groups(
       outgoing_group="input",
       incoming_group="hidden",
       link_type="input_links",
       rand_mean=-1.0,
       rand_range=0.5
   )

This will create a full projection, with link weights randomly initialized to
values in the interval ``[-1.5, -0.5]``.  

**2. Create a bidirectional random projection between "in" and "hid":**

.. code-block:: python

   net.connect_groups(
       outgoing_group="in",
       incoming_group="hid",
       link_type="t1",
       proj_type="random",
       rand_mean=0.0,
       rand_range=0.2,
       bidirectional=True
   )

In this example, the reverse projection (from ``"hid"`` to ``"in"``) is also created automatically.

See Also
--------

- :doc:`add_group`: |add_group|
- :doc:`connect_group_to_unit`: |connect_group_to_unit|
- :doc:`connect_units`: |connect_units|
