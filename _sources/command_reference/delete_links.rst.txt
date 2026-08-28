.. _delete_links:

delete_links
=============

.. the description is stored in the conf.py file

|delete_links|

Usage
-----

.. code-block:: python

   net.delete_links(
       link_type=None
   )


Description
-----------

``delete_links`` removes all links of the specified type. If ``link_type`` is ``None``, all links in the network are removed.

The function returns the number of links removed (as an integer).

Arguments
---------

- **link_type** (``str`` | ``None``, default ``None``): Type or label assigned to the links.

Examples
--------

**1. Delete all links in the network**

.. code-block:: python

   net.delete_links(
       link_type=None
   )

**2. Delete only one link type**

.. code-block:: python

   net.delete_links(
       link_type="uniform"
   )

See Also
--------

- :doc:`delete_group_inputs`: |delete_group_inputs|
- :doc:`delete_group_outputs`: |delete_group_outputs|
- :doc:`delete_unit_inputs`: |delete_unit_inputs|
- :doc:`delete_unit_outputs`: |delete_unit_outputs|
