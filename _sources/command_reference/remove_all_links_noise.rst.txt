.. _remove_all_links_noise:

remove_all_links_noise
======================

Removes noise from all links in the network.

Usage
-----

.. code-block:: python

   net.remove_all_links_noise()

Description
-----------

``remove_all_links_noise`` removes any previously applied noise from all links in the network. This function ensures that all incoming links across all groups are restored to their original state before noise was added.

This function is useful for **resetting** network weights after applying noise-based perturbations.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Remove noise from all links in the network:**

.. code-block:: python

   net.remove_all_links_noise()

See Also
--------

- :meth:`add_all_links_noise <your_module.add_all_links_noise>`: Applies noise to all links in the network.
- :meth:`add_links_specific_noise <your_module.add_links_specific_noise>`: Applies noise to specific links between two groups.
- :meth:`remove_links_specific_noise <your_module.remove_links_specific_noise>`: Removes noise from specific links between two groups.
