.. _remove_links_specific_noise:

remove_links_specific_noise
===========================

Removes noise from specific links between two groups in the network.

Usage
-----

.. code-block:: python

   net.remove_links_specific_noise(outgoing_group, incoming_group)

Description
-----------

``remove_links_specific_noise`` removes any noise that has been applied to links between the ``specified outgoing_group`` and ``incoming_group``. This function is useful when you need to reset or clean up connections that were previously affected by noise.

Each link within the ``incoming_group``’s set of incoming connections will be processed, and noise will be removed.

Arguments
---------

- **outgoing_group** (``str``): The name of the group that sends data forward.

- **incoming_group** (``str``): The name of the group receiving the links.

Examples
--------

**1. Remove noise from links between "hidden1" and "output":**

.. code-block:: python

   net.remove_links_specific_noise("hidden1", "output")

**2. Remove noise from connections between "input" and "hidden":**

.. code-block:: python

   net.remove_links_specific_noise("input", "hidden")

See Also
--------

- :meth:`links_specific_lesion <your_module.links_specific_lesion>`: Lesions specific links between two groups.
- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions all links in the network.