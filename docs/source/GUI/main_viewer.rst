Main Viewer
================

The Main Viewer contains 8 panels: network information, file loading, display, training algorithms, training hyperparameters, training options, training controls, and an exit button.

.. figure:: ../images/main-viewer-empty.png
   :alt: The Main Viewer window. Only the run script, load examples and exit button are enabled at startup.
   :width: 600
   :align: center

Most buttons will be disabled because a network has not been loaded yet. To begin exploring the GUI, load a network python script using the ``Run Script`` button. Many example networks exist in the ``examples/example_networks/`` folder of PyLens. For instance, to familiarize yourself with the GUI functionality, you could load ``encoder_example_one.py`` from ``encoder_examples/``.

.. figure:: ../images/run-script.png
   :alt: Click the run-script button to load a network from a python file.
   :width: 600
   :align: center

The GUI buttons should now be enabled. The training hyperparemeters are populated by those specified in the network's python script.

Next, we will step through the GUI panels individually.


Network Information
----------------------
The first panel, the network info panel, shows the name of the loaded network and its training and testing example sets. Clicking on ``Training Set`` or ``Testing Set`` will reveal a pull-down menu that lets you change the example sets.

.. figure:: ../images/training-set.png
   :alt: Training sets loaded in encoder_example_one.
   :width: 600
   :align: center

File Loading
--------------
This panel provides shortcuts for executing some common actions, including running scripts, loading example sets, and saving and loading weights. Each button will cause a file browser to be opened to let you choose the file for reading or writing. Depending on the command, another pop-up may appear once the file is chosen for setting any remaining options. 

Network Display
--------------------
This panel provides access to three graphical displays: the ``Unit Viewer``, the ``Link Viewer`` and ``New Graph``. They provide graphical ways to probe network properties such as unit outputs, weights connecting different layers, and error over time.

See here for details about each graphical display:

- :doc:`Unit Viewer <unit_viewer>`
- :doc:`Link Viewer <link_viewer>`
- :doc:`New Graph <creating_graphs>`

Training Algorithms
--------------------
This panel contains radio buttons for selecting the weight update method. The selected algorithm will be used to optimize the weights when training the network.

.. figure:: ../images/steepest.png
   :alt: The steepest descent algorithm is selected and will be used for training. 
   :width: 600
   :align: center

Training Hyperparameters
--------------------------
This panel provides access to commonly used training hyperparameters. When loading a network, the entry boxes will be populated with the default hyperparameters specified in the network's python script. To change a value, click in the entry box, edit the value, and then press Enter.

.. figure:: ../images/hyper-params.png
   :alt: Default training hyperparameters for the encoder_example_one network. 
   :width: 600
   :align: center

Training Options
-----------------
This panel allows users to specify key training options. To change a value, click in the entry box, edit the value, and then press Enter.

The training options are:

- **Batch Size**: The size of the batch seen at each training step. If 0, the batch size will include all examples in the Example Set.

- **Error Criterion**: The error threshold at which to stop training.

- **Weight Updates**: Number of weight updates that will be executed for a single press of "Start Training".

- **Report Interval**: Number of weight updates between training stats reports and graph updates.

- **Parallel Training**: Enable parallel training mode that utilizes multiple cores of the CPU

- **Number of Cores**: Number of CPU cores used in parallel training mode.

.. figure:: ../images/training-options.png
   :alt: Default training hyperparameters for the encoder_example_one network. 
   :width: 600
   :align: center

Training Control
------------------
Once a network and training examples are loaded, clicking the ``Start Training`` button will begin training the network. Training can be stopped with the ``Stop Training`` button. The ``Reset Network`` button will reinitialize the networks weights and parameters so that training can begin from scratch. ``Test Network`` will run the network on the set of testing examples, if a Testing Set is loaded.

.. # explanations missing for ``Reset Training Set`` and ``Record Outputs``

.. figure:: ../images/training-control.png
   :alt: Training control buttons. 
   :width: 600
   :align: center

Exit
----------

Clicking on ``Exit`` closes the GUI. Alternatively, you can click the "x" button at the top-right corner of the control panel.

.. # ``Wait`` button is not explained

