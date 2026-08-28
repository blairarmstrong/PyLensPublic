Link Viewer
================
The Link Viewer provides a graphical illustration of the links between the layers of the network.

The Link Viewer can be invoked from the GUI's Main Viewer once a network is loaded. As an example, here we load the network python script ``encoder_example_one.py`` from the ``examples/example_networks/encoder_examples/`` path within PyLens. For this network, the Link Viewer looks like this:

.. figure:: ../images/link-viewer.png
   :alt: blank image of the link viewer
   :width: 600
   :align: center

The Link Viewer consists of:

- A **main panel** displaying a two-dimensional connectivity matrix of network links. The rows specify the layers from which the links are leaving, while the columns specify the receiving layers.
- An **information bar** (in the upper half) that displays summary statistics of all the displayed links and information about the currently selected link.
- A **drop-down menu** (at the very top of the window (on Windows) or the screen (on Mac)) that allows you to control miscellaneous settings.

Let's step through the key functions of the Link Viewer.

To **display information about a specific link**, hover over its cell. The boxes on the right of the information bar show the value of the link as well as the group names and indexes of the units it connects.

To **change the type of cell value** to display, use the ``Value`` tab. The options are ``Link Weight``, ``Link Derivs`` and ``Link Deltas``.

To **update the link's values** as training progresses, click on the ``Viewer`` tab at the top and specify the ``Update After`` interval.

.. figure:: ../images/link-viewer-update-after.png
   :alt: Options for updating interval of the weights.
   :width: 600
   :align: center

The **size and spacing of the link cells** can be modified by clicking on the ``Viewer`` tab and selecting ``Cell Size`` or ``Cell Spacing``. 

To **hide or re-display link groups** in the Link Viewer, click on the ``Sending Groups`` or ``Receiving Groups`` tabs and select the group to hide or redisplay.

.. figure:: ../images/link-viewer-hide-input.png
   :alt: Hiding the input group's outgoing links.
   :width: 600
   :align: center

.. figure:: ../images/link-viewer-input-hidden.png
   :alt: Hiding the input group's outgoing links.
   :width: 600
   :align: center

The ``Palette`` tab includes **different color palettes** you can use to display the links.

Same as in the Unit Viewer, the ``Viewer`` tab also contains options to ``Update`` the colors of the link cells or to ``Refresh`` (i.e., rebuild the representation of) the links.


