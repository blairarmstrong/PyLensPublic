.. _polarity:

polarity
=========

.. the description is stored in the conf.py file

|polarity|

Usage
-----

.. code-block:: python

   net.polarity(
       action,
       group_names="*"
   )

Description
-----------

``polarity`` takes an action and a list of groups on which to perform that action.

Possible **actions**:

- ``reset``: The groups will erase all of the stored polarity information.
- ``update``: The current polarity of each of the units in the group will be computed and these observations are recorded in the group structure using the ``polarity_sum`` and ``polarity_num`` fields. It is common to add a polarity ``update`` call to the ``postEventProc`` or ``postExampleProc`` so that the polarity is tracked following each event or example.
- ``report``: The command will print a dictionary mapping each group name to its average polarity since the last reset.

**Polarity** is a measure of the degree to which the units in a group are adopting outputs close to the minimum or maximum possible output value. In a logistic group, the minimum output is 0 and the maximum is 1, but other types of units may have a different range. In order to compute the polarity of a unit, its output is first normalized to the range [0,1].

The polarity of a unit whose normalized output is x is given by the following formula:

.. note::

    polarity = x log2(x) + (1-x) log2(1-x) + 1
    
This is a U-shaped function. If the normalized output is 0.5, the polarity is 0. As the output increases towards 1 or decreases towards 0, the polarity grows towards 1.

The average polarity of the units in a group, across several training or testing examples, is a useful measure. A high polarity, close to 1, indicates that the units are adopting binary states. A low polarity can indicate that there are too few units to solve the given task and the network is forced to make use of intermediate activation values. In the absence of noise, it is also possible for a network to find a low-polarity solution when it has too many units.

Various techniques can be used to encourage a network to adopt more polarized output values. Indirectly this can be done by adding noise to the unit inputs. Units with non-polarized values will be more susceptible to this kind of noise and the network will be more successful with a solution that uses binary activations. An output cost function, such as LOGISTIC_COST, can also be used to directly encourage polarized output values. In fact, the LOGISTIC_COST function uses a cost function that is simply the inverse of polarity.

Arguments
---------

- **action** (``str``): The action to be performed on polarity values. One of: ``"reset"``, ``"update"``, ``"report"``.

- **group_names** (``str`` or ``list``, optional, default ``"*"``): A list of group names on which the action is performed. Use ``"*"`` or omit to apply to all groups. A single group name string is also accepted.

Examples
--------

**1. Reset polarity tracking, then update after processing**

.. code-block:: python

   net.polarity("reset")
   # ... run examples ...
   net.polarity("update")

**2. Report average polarity for specific groups**

.. code-block:: python

   net.polarity("report", group_names=["hidden", "output"])

See Also
--------

- :doc:`get_group_type`: |get_group_type|