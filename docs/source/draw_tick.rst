.. _draw_tick:

draw_tick
=========

Draws the network's layers for the current tick.

Usage
-----

.. code-block:: python

   net.draw_tick()

Description
-----------

``draw_tick`` visually **renders the current state** of the network at the current tick. This function:

- **Draws all groups** in the network, including:

  - **Input layers**
  - **Hidden layers**
  - **Output layers**
  - **Bias and target layers**

- **Manages visual effects**:

  - Resets **highlighting effects** if ``link_flag`` is disabled.
  - Applies **weight coloring** if ``link_flag`` is enabled.

This function is useful for **debugging**, **visualizing network updates**, and **observing real-time changes** in activations and weights.

Arguments
---------

This function does not take any arguments.

Examples
--------

**1. Draw the current network state at a tick:**

.. code-block:: python

   net.draw_tick()

See Also
--------

- :meth:`draw_group <your_module.draw_group>`: Draws a specific group within the network.
- :meth:`color_weights <your_module.color_weights>`: Applies weight coloring when in link mode.
- :meth:`update_canvas <your_module.update_canvas>`: Refreshes the visualization canvas.
