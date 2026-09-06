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

.. autoclass:: PyLens.backend.network.Network

Simple Recurrent Backpropagation Through Time Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyLens.backend.srbptt_network.SRBPTTNetwork

Continuous Network
~~~~~~~~~~~~~~~~~~

.. autoclass:: PyLens.backend.continuous_network.ContinuousNetwork

Boltzmann Machine
~~~~~~~~~~~~~~~~~

.. autoclass:: PyLens.backend.boltzmann_network.BoltzmannMachine



Groups
^^^^^^

This is the object representing a group (layer) of units.

.. autoclass:: PyLens.backend.group.Group


Input transformations
^^^^^^^^^^^^^^^^^^^^^

This is the object that applies an input transformation to a group of units.

.. autoclass:: PyLens.backend.inputs.input_transform.Input_Transform

Output transformations
^^^^^^^^^^^^^^^^^^^^^^

This is the object that applies an output transformation from a group of units, also called an activation function.

.. autoclass:: PyLens.backend.output_transforms.basic.Basic




Links
^^^^^

This is the object representing the links (weights) between two groups (layers) of units.

.. autoclass:: PyLens.backend.link.link.Link

Fully connected links
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyLens.backend.link.link_full.LinkFull


One-to-one connected links
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyLens.backend.link.link_one_to_one.LinkOneToOne

Randomly connected links
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyLens.backend.link.link_random.LinkRandom


Optimizers
^^^^^^^^^^

This is the object representing the weights between two groups (layers) of units.

.. autoclass:: PyLens.backend.optimizer.optimizer.Optimizer


Example Classes
---------------

There are three example class objects: example set, example, and event. The example iterator iterates over the examples in an example set.


Example Set
^^^^^^^^^^^

.. autoclass:: PyLens.examples.example_set.ExampleSet


Example
^^^^^^^

.. autoclass:: PyLens.examples.example.Example


Event
^^^^^

.. autoclass:: PyLens.examples.event.Event


Example Iterator
^^^^^^^^^^^^^^^^

.. autoclass:: PyLens.examples.example_iterator.ExampleIterator


GUI Classes
-----------


Main viewer
^^^^^^^^^^^

.. autoclass:: PyLens.gui.main_viewer_tk.main_viewer_tk


Unit viewer
^^^^^^^^^^^

.. autoclass:: PyLens.gui.unit_viewer_tk.FrameExamplesProgram

Link viewer
^^^^^^^^^^^

.. autoclass:: PyLens.gui.link_viewer_tk.link_viewer

Graph viewer
^^^^^^^^^^^^

.. autoclass:: PyLens.gui.graph_viewer_tk.GraphViewer
