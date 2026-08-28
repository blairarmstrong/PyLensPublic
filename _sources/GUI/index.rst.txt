Graphical User Interface (GUI)
=================================

The PyLens GUI allows users to load and train networks while also providing a graphical interface to explore how the network weights, units and performance evolve over training updates. To start an empty PyLens GUI, install PyLens and run the following code from the root folder.

.. code-block:: console

    python examples/start_gui.py

The GUI can also be opened from a PyLens script by adding the following line. ``simulator`` is the simulator object. ``net`` is the neural network object. See more in :doc:`Coding Tutorial <../coding_tutorial>`. 

.. code-block:: python

    simulator.use_gui(net)

See the sections below for more information on how to use the GUI and its subcomponents.

.. toctree::
   :maxdepth: 2

   main_viewer
   unit_viewer
   link_viewer
   creating_graphs
