Creating Graphs
================
A range of network properties can be plotted during training. To illustrate these functions, we load the network python script ``encoder_example_one.py`` from the ``examples/example_networks/encoder_examples/`` folder of PyLens.

Click the ``New Graph`` button in the Main Viewer to open a new window with an empty graph. Move the window next to the Main Viewer window so that you can access both. Now click on ``Start Training`` in the Main Viewer.

By default, the network's error will be plotted. The error will be plotted at each Report Interval for the number of training steps specified in the Main Viewer.

.. figure:: ../images/error-plot.png
   :alt: Error plotted every 10 update intervals for 100 training steps.
   :width: 600
   :align: center

The toolbar at the top of the graph window includes the default matplotlib controls: resetting the view, panning, zooming, configuring the graph's size, and saving.

.. figure:: ../images/graph-controls.png
   :alt: Graph controls.
   :width: 600
   :align: center

Next to the ``New Graph`` button in the Main Viewer is a text entry box that can be used to select which network property to plot. Arbitrary network attributes (with numerical values) can be plotted by specifying the attribute name on the text entry box. For example, to plot the output of the first output unit in the last layer of the encoder_example_one network, you would type ``output_groups[-1].output_matrix[1]``.

.. figure:: ../images/output-unit-plot.png
   :alt: Plotting the output unit. 
   :width: 600
   :align: center

To compare training progress between different runs, you can press the ``Reset Network`` button in the Main Viewer while plotting. This starts a new trace in the graph. The latest trace will be plotted in black color.

.. figure:: ../images/traces.png
   :alt: Error plot for different training runs.
   :width: 600
   :align: center

Multiple graphs can be invoked by re-clicking the New Graph button. 

