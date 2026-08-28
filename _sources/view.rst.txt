.. _view:

View
====

Opens the main control display.

Usage
-----

.. code-block:: python

    sim = Simulator()
    net = sim.create_net(name="network_name")
    sim.use_gui(net)

Description
-----------

This function opens the **main control display** in PyLens. If the display is already open, it will bring it to the front.

Calling ``sim.use_gui(net)`` enables visualization of the specified network ``net`` using the built-in GUI.

``use_gui`` allows users to **view network components, monitor link weights, and analyze unit activations** in real time.

See Also
--------

- :ref:`graphObject` - Opens a graph for tracking values over time.
- :ref:`viewLinks` - Opens the link viewer for inspecting connections.