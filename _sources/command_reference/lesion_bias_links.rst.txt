.. _lesion_bias_links:

lesion_bias_links
=================

Applies a lesion to the bias links in the network.

Usage
-----

.. code-block:: python

   net.lesion_bias_links(group_name=None, lesion_rate=0)

Description
-----------

``lesion_bias_links`` selectively **lesions** (disables or removes) bias links in the network.

This function is useful for testing the impact of bias removal in different network configurations.

Arguments
---------

- **group_name** (``str``, optional, default ``None``): The name of the group whose bias link should be lesioned. If not provided, bias links in **all groups** will be lesioned.

- **lesion_rate** (``float``, optional, default ``0``): The probability (between 0 and 1) that each bias link will be lesioned.

Examples
--------

**1. Lesion the bias link for the "hidden1" group with a 50% probability:**

.. code-block:: python

   net.lesion_bias_links(group_name="hidden1", lesion_rate=0.5)

**2. Lesion the bias links for all groups in the network:**

.. code-block:: python

   net.lesion_bias_links(lesion_rate=1.0)

See Also
--------

- :meth:`add_group <your_module.add_group>`: Adds a group to the network and sets up bias projections.
- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions all links in the network.
- :meth:`heal_all_groups <your_module.heal_all_groups>`: Heals all lesioned groups in the network.
