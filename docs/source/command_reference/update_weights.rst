.. _update_weights:

update_weights
==============

.. the description is stored in the conf.py file

|update_weights|

Usage
-----

.. code-block:: python

   net.update_weights(report_request=False)

Description
-----------

``Network.update_weights`` applies **one optimizer step** using the **accumulated link derivatives** from the last forward/backward pass. It delegates to ``net.optimizer.update_weights``.

This matches the idea of cLens ``updateWeights`` (single weight update from current derivatives), but PyLens has **no** Tcl-style flags such as ``-algorithm`` or ``-noreset``. Use :doc:`set_update_method` to choose steepest, momentum, Adam, etc. Control derivative resets with your training loop (see :doc:`train`) or ``net.reset_derivs()`` when building a custom pattern.

Arguments
---------

- **report_request** (``bool``, optional, default ``False``): If ``True``, the active optimizer may gather extra statistics for progress reporting.

Examples
--------

**1. Single update after a custom forward/backward:**

.. code-block:: python

   net.forward()
   net.backward()
   net.update_weights()

**2. Update with reporting hooks enabled:**

.. code-block:: python

   net.update_weights(report_request=True)

See Also
--------

- :doc:`train`: |train|
- :doc:`set_update_method`: |set_update_method|
