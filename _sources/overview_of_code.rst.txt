Overview of the Code
=====================
This page describes the structure of the most important class objects used in PyLens. PyLens includes three major sets of objects. The first set is the network objects, which include the neural network object and other objects responsible for neural network representation, computation, and training. The second set is the example objects, which represent training and testing example sets in a structure suitable for cognitive modeling. The third set is the GUI objects, which visualize the neural network units, layers, weights, and graphs, and provide command buttons.


Network Classes
---------------

The `backend` contains network, groups (layers), links (weights), and
optimizer classes.



Network
^^^^^^^

This is the backbone neural network object. It connects other components, such as groups (layers), input/output transformations, and links (weights).

.. autoclass:: src.backend.network.Network

Simple Recurrent Backpropagation Through Time Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.backend.srbptt_network.SRBPTTNetwork

Continuous Network
~~~~~~~~~~~~~~~~~~

.. autoclass:: src.backend.continuous_network.ContinuousNetwork

Boltzmann Machine
~~~~~~~~~~~~~~~~~

.. autoclass:: src.backend.boltzmann_network.BoltzmannMachine



Groups
^^^^^^

This is the object representing a group (layer) of units.

.. autoclass:: src.backend.group.Group


Input transformations
^^^^^^^^^^^^^^^^^^^^^

This is the object that applies an input transformation to a group of units.

.. autoclass:: src.backend.inputs.input_transform.Input_Transform

Output transformations
^^^^^^^^^^^^^^^^^^^^^^

This is the object that applies an output transformation from a group of units, also called an activation function.

.. autoclass:: src.backend.output_transforms.basic.Basic




Links
^^^^^

This is the object representing the links (weights) between two groups (layers) of units.

.. autoclass:: src.backend.link.link.Link

Fully connected links
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.backend.link.link_full.LinkFull


One-to-one connected links
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.backend.link.link_one_to_one.LinkOneToOne

Randomly connected links
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.backend.link.link_random.LinkRandom


Optimizers
^^^^^^^^^^

This is the object representing the weights between two groups (layers) of units.

.. autoclass:: src.backend.optimizer.optimizer.Optimizer


Example Classes
---------------

There are three example class objects: example set, example, and event. The example iterator iterates over the examples in an example set.


Example Set
^^^^^^^^^^^

.. autoclass:: src.examples.example_set.ExampleSet


Example
^^^^^^^

.. autoclass:: src.examples.example.Example


Event
^^^^^

.. autoclass:: src.examples.event.Event


Example Iterator
^^^^^^^^^^^^^^^^

.. autoclass:: src.examples.example_iterator.ExampleIterator


GUI Classes
-----------


Main viewer
^^^^^^^^^^^

.. autoclass:: src.gui.main_viewer_tk.main_viewer_tk


Unit viewer
^^^^^^^^^^^

.. autoclass:: src.gui.unit_viewer_tk.FrameExamplesProgram

Link viewer
^^^^^^^^^^^

.. autoclass:: src.gui.link_viewer_tk.link_viewer

Graph viewer
^^^^^^^^^^^^

.. autoclass:: src.gui.graph_viewer_tk.GraphViewer
