.. _set_update_method:

set_update_method
=====================

.. the description is stored in the conf.py file

|set_update_method|

Usage
-----

.. code-block:: python

    net.set_update_method(
        lr, 
        update_method="steepest"
    )

Description
-----------

``set_update_method`` configures the **weight update method** for optimizing network parameters. The string you pass is normalized with ``.lower()`` and must match one of the spellings below (these are the Python API names; cLens documentation sometimes uses forms like ``dougsMomentum`` or ``deltaBarDelta`` — in PyLens use the quoted strings).

The function **instantiates the matching optimizer** on the network. If ``update_method`` is not recognized, PyLens falls back to the **global network default** (``PAR_N_algorithm``).

``"steepest"`` (default) — Pure gradient descent in weight space (no momentum from prior steps).

``"momentum"`` — Each weight change blends the current gradient step with a fraction of the previous step, controlled by the optimizer’s momentum parameter. The effective step size can grow roughly like ``1 / (1 - momentum)`` compared to plain steepest descent.

``"dougs momentum"`` — Like standard momentum, but the **pre-momentum** weight step vector is **capped** so its length does not exceed 1.0. After adding momentum, the combined update can still grow up to about ``1 / (1 - momentum)``. This often allows **larger stable learning rates** early in training than plain momentum.

``"delta bar delta"`` — Per-link adaptive learning rates. Each link keeps a multiplier (see the optimizer / link fields such as ``link_learning_rate``). When consecutive updates for a weight move in a consistent direction vs. when they reverse, the multiplier is adjusted using the optimizer’s ``rate_increment`` and ``rate_decrement`` (see :doc:`../overview_of_code` and optimizer parameters). The base ``lr`` you pass to ``set_update_method`` still scales the overall step.

``"adam"`` — Adam optimizer (PyTorch path uses the corresponding torch implementation where applicable).

Arguments
---------

- **lr** (``float``): The learning rate used for the optimizer.

- **update_method** (``str``, optional, default ``"steepest"``): The weight update method to use. Options: ``"steepest"``, ``"momentum"``, ``"dougs momentum"``, ``"delta bar delta"``, ``"adam"``.

Examples
--------

**1. Steepest descent (default), common for small demos:**

.. code-block:: python

   net.set_update_method(0.5, "steepest")

**2. Momentum with a learning rate of 0.01:**

.. code-block:: python

   net.set_update_method(lr=0.01, update_method="momentum")

**3. Doug’s momentum (bounded pre-momentum step):**

.. code-block:: python

   net.set_update_method(0.2, "dougs momentum")

**4. Delta–bar–delta (per-weight learning rates):**

.. code-block:: python

   net.set_update_method(0.1, "delta bar delta")

**5. Adam:**

.. code-block:: python

   net.set_update_method(lr=0.001, update_method="adam")

**6. Invalid method name — falls back to** ``PAR_N_algorithm`` **:**

.. code-block:: python

   net.set_update_method(lr=0.1, update_method="invalid_method")

See Also
--------

- :doc:`train`: |train|
