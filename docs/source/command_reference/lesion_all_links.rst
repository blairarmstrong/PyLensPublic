.. _lesion_all_links:

lesion_all_links
================

Removes or modifies incoming links to all groups in the network.

Usage
-----

.. code-block:: python

   net.lesion_all_links(lesion_rate)

Description
-----------

``lesion_all_links`` applies a lesioning process to all incoming links in every group within the network. The lesioning is applied based on a specified **lesion rate**, which determines the probability that each link will be affected.

By default:

- The lesioning process **removes** affected links entirely.

- If a different behavior (such as modifying weights instead of deletion) is needed, this should be handled in the ``lesion_link`` method of the link object.

Since lesioning is irreversible, consider saving the network state (e.g., saving weights) before applying this function.

If reproducibility is required, ensure the random seed is set before calling ``lesion_all_links``, so that the same links are lesioned in repeated runs.

Arguments
---------

- **lesion_rate** (``float``):
  A probability value between 0 and 1 representing the chance that each link will be lesioned (removed or modified). For example:
  
  - ``1.0`` → All links will be lesioned.
  - ``0.5`` → Roughly 50% of links will be affected.
  - ``0.0`` → No links will be lesioned.

Examples
--------

**1. Remove 50% of all incoming connections from every group in the network**:

.. code-block:: python

   net.lesion_all_links(lesion_rate=0.5)

**2. Remove all links entirely**:

.. code-block:: python

   net.lesion_all_links(lesion_rate=1.0)

See Also
--------