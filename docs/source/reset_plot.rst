.. _reset_plot:

reset_plot
==========

Clears all elements from the given canvas.

Usage
-----

.. code-block:: python

   net.reset_plot(canvas)

Description
-----------

``reset_plot`` removes all **drawn elements** from the provided canvas, resetting the visualization. This function:

- **Deletes all elements** from the canvas.
- Ensures a **clean slate** for redrawing plots or network visualizations.

Arguments
---------

- **canvas** (``Canvas``): The canvas object that needs to be cleared.

Examples
--------

**1. Reset a visualization canvas:**

.. code-block:: python

   net.reset_plot(my_canvas)

See Also
--------

- :meth:`draw_tick <your_module.draw_tick>`: Draws network layers for visualization.
- :meth:`update_canvas <your_module.update_canvas>`: Refreshes the visualization canvas.
- :meth:`reset_network <your_module.reset_network>`: Resets the network state, including visualization elements.
