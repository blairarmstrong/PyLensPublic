.. _add_group:

add_group
=========

.. the description is stored in the conf.py file

|add_group|

Usage
-----

.. code-block:: python

   net.add_group(
       num_units,
       name=None,
       group_type="hidden",
       input_transforms=None,
       output_transforms=None,
       error_function=None,
       lesion_rate=None,
       dropout_rate=None,
       num_cols=None,
       biased=None,
       unit_cost_function=None
   )

Description
-----------

``add_group`` creates a group with the specified name and number of units and places it at the end of the network’s internal list of groups. The group must contain at least one unit.

If no **name** is provided, one is generated automatically using the pattern ``<group_type>_<index>``. The units in the group may also be named or tracked internally.

By default:

- **hidden** groups use a ``"dot"`` input transform and a ``"sigmoid"`` output transform.
- **output** groups use a ``"dot"`` input transform and a ``"sigmoid"`` output transform (though you can override this).
- **input** groups have empty input/output transform lists (i.e., no transforms unless you specify them).

Ordinarily, all *hidden* and *output* groups receive bias inputs (if a bias group exists), while *input* and *bias* groups do not. You can override this behavior with the *biased* argument if needed.

If the group is an *output* group and no *error_function* is specified, a default error function may be selected based on the final output transform. For example, a *sigmoid* output transform might default to a cross-entropy error. Likewise, a *linear* output transform might default to a mean-squared error.

Use *unit_cost_function* for any additional cost or penalty you want to apply at the level of each individual unit (e.g., to encourage sparsity).

.. note::

   If you have a dedicated bias group in the network, a projection from this bias group to all newly created *hidden* or *output* groups will be formed automatically unless ``biased=False``.

Arguments
---------

- **num_units** (``int``): Number of units in the group. Must be at least 1.

- **name** (``str``, optional): Name for the group. If omitted, a default name is generated (e.g., ``"hidden_0"``, ``"hidden_1"``, etc.).

- **group_type** (``str``, optional): Determines the kind of group. Possible values typically include:
  
  - ``"input"``
  - ``"hidden"`` (default)
  - ``"output"``
  - ``"bias"``
  - ``"elman"`` 

- **input_transforms** (``list``, optional): A list of input transform names. Common values include:
  
  - ``"dot"``
  - ``"product"``
  - ``"soft_clamp"``
  - ``"in_copy"``
  - ``"in_integr"``
  - ``"distance"``
  - ``"boltzmann"``
  
  If unspecified, defaults are chosen based on ``group_type``.

- **output_transforms** (``list``, optional): A list of output transform names. Common values include:
  
  - basic transform: ``"linear"``, ``"sigmoid"``, ``"tanh"``, ``"soft_max"``
  - clamping: ``"hard_clamp"``, ``"noise"``, ``"cropped"``, ``"gaussian"``
  - modifier: ``"exponential"``, ``"elman_clamp"``, ``"out_integr"``, ``"out_copy"``
  
  If unspecified, defaults are chosen based on ``group_type``.

- **error_function** (optional):
  The function used to compute the error for *output* groups. Typical options might be:
  
  - ``"squared"`` for sum-of-squares error
  - ``"cross_entropy"`` for cross-entropy error
  - ``"cosine"`` for cosine error which calculates the 1.0 - the cosine of the angle between the output and target vectors. 
  - ``"divergence"`` for Kullback–Leibler divergence
  
  If omitted, the system may pick a default based on ``output_transforms``.

- **lesion_rate** (``float``, optional):
  Rate at which units are disabled or “lesioned.” A value of 0 means no lesioning.

- **dropout_rate** (``float``, optional):
  Probability of dropping out a unit’s output during training. Similar to a standard dropout technique.

- **num_cols** (``int``, optional):
  Number of columns in the group if you want to organize the group’s units in a grid-like structure. This might not be used in some frameworks.

- **biased** (``bool``, optional):
  Whether this group should receive bias inputs from the network’s bias group. If not specified, defaults depend on the ``group_type``:
  
  - ``True`` for hidden or output
  - ``False`` for input, elman, or bias groups

- **unit_cost_function** (optional):
  A function to compute an additional penalty or cost on each unit’s activation. No default; if unspecified, no per-unit cost is applied.

  - ``"conv_quad"`` 
  - ``"cosine"``
  - ``"linear"`` 
  - ``"logistic"`` 
  - ``"quadratic"``

Examples
--------

**1. Create a hidden layer with dot-product input and linear output:**

.. code-block:: python

   net.add_group(
       num_units=100,
       name="myGroup",
       group_type="hidden",
       input_transforms=["dot"],
       output_transforms=["linear"],
       biased=True
   )

**2. Create an output layer with explicit transforms and squared error:**

.. code-block:: python

   net.add_group(
       num_units=10,
       name="group_2",
       group_type="output",
       input_transforms=["dot"],
       output_transforms=["exponential"],
       error_function="squared",
       biased=False
   )

See Also
--------

- :doc:`connect_groups`: |connect_groups|
- :doc:`order_groups`: |order_groups|
- :doc:`delete_group`: |delete_group|
