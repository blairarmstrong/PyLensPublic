Coding Tutorial
==========================================

Apart from loading and running already existing scripts, you can **write your own coding scripts** to create networks with PyLens. This tutorial provides an overview of how to write scripts for PyLens. For details about the commands used in this tutorial, see the :doc:`Command Reference<command_reference/index>`.

As an example, we will create **a network that learns to recognize digits**. In our training data, each input consists of a picture of a digit simplified into a 4x5 grid of 1s (meaning that a pixel is "filled") and 0s (meaning that a pixel is "empty").

For instance, the handwritten digit 5 could be represented like this:

.. figure:: images/digit-5.png
   :alt: an image of a handwritten digit 5 and its representation as a grid of 1s and 0s
   :width: 180
   :align: center

Our goal is to train a network to produce the correct label (i.e., output representation) for each input. Here, we implement this with two hidden layers to illustrate the basic functionality of creating and connecting layers. (However, you could probably solve the problem without hidden layers.)

Preparation
--------------

1. Create an **empty python script called** ``digits.py``.

We will now gradually build the script that performs digit recognition.

Using the Simulator
-------------------------------------

Start your script with the following line to import the **simulator** module, which contains the PyLens simulator engine that links all the deployed neural networks and graphical user interfaces.

.. code-block:: python

   from PyLens.simulator import Simulator

Then, create a simulator for the current network.

.. code-block:: python

   sim_one = Simulator(name="simulator")

Creating Network Layers
-------------------------------------

Next, we specify the network structure by following these steps:
 
1. **Create a new network** named ``digits_net``. By default, the network contains a **bias** group consisting of a single bias unit, which (again by default) will be connected to each unit in the hidden and output layers.

.. code-block:: python

    digits_net = sim_one.create_net(name='digits')

2. **Add an input layer** (or group) consisting of 20 units. The ``num_cols`` argument accounts for the fact that the input is represented in a grid-like structure with four columns, as explained above.
    
.. code-block:: python

    digits_net.add_group(
        20,
        name="input",
        group_type="input",
        num_cols=4,
    )

3. **Add one hidden layer** with 20 units each. By default, hidden groups use a ``dot`` input transform and a ``sigmoid`` output transform.
    
.. code-block:: python

    digits_net.add_group(
            20,
            name="hidden",
            group_type="hidden",
    )

4. Finally, **add an output layer** consisting of 3 units. Again by default, the output group uses a ``dot`` input transform and a ``sigmoid`` output transform. Based on the output transform, the network will automatically select an error function. Here, it defaults to ``cross-entropy`` error.

.. code-block:: python

    digits_net.add_group(
        3,
        name="output",
        group_type="output",
    )

Connecting the Layers
-------------------------------------

We next create links between the layers by following these steps:

1. **Connect the input layer to the hidden layer**.    

.. code-block:: python

    digits_net.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
    )

2. **Connect the hidden layer to the output layer**, again using full projection.
    
.. code-block:: python

    digits_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
    )

Loading an Example Set
-------------------------------------

After specifying the network structure, we load the training set, which is located in the ``./examples/lens_example_input/`` folder.

.. code-block:: python

    digits_net.load_example_set("./examples/lens_example_input/digits.ex", "digits")

Training the Network
-------------------------------------

Now, all that is left to do is to **set the training parameters**, including the learning rate ``lr`` and the ``update_method``.

.. code-block:: python

    digits_net.set_update_method(
        lr = 0.1, 
        update_method = "adam"
    )

Then, **we train the network** for 200 epochs (or updates). We set ``batch_size`` to 0 to ensure that the network is trained on all examples during each update. The ``report_interval`` argument specifies the number of updates between progress reports.

.. code-block:: python

    digits_net.train(
        100, 
        batch_size = 0, 
        report_interval = 10
    )

**Optionally**, we can use **parallel training** to speed up training by splitting it over multiple processing units (or workers). To enable parallel training, switch the ``parallel_mode`` flag to ``True`` (for details, see :doc:`Parallel Training<parallel>`).

.. code-block:: python

    digits_net.train(
        100, 
        batch_size = 0, 
        parallel_mode = True
    )

Running the Script
-------------------------------------

**To run the script**, open an Anaconda Prompt, activate your virtual environment (optional), and navigate to your **PyLens root folder** (for details, see the :doc:`Beginner's tutorial<beginner_tutorial>`).

.. code-block:: bash

    python digits.py

Creating Plots
--------------------------------------------------------------
.. or: Plotting Network Error and Unit Activation

Currently, PYLens does not provide a direct Python API for plotting. All graphs must be created through the GUI.

Add this code line at the end of your script to open the GUI:

.. code-block:: python

    sim_one.use_gui(digits_net)

In the GUI, you can click on the ``New Graph`` button to create plots, and you can use the text entry box next to it to specify which network property to plot (e.g., error). For details, see :doc:`Creating Graphs<GUI/creating_graphs>`.

Full Scripts
--------------------------------------------------------------

Full ``digits.py`` script is provided at ``examples/example_networks/digits_examples/digits.py``

Where to Go From Here?
-------------------------

For next steps, you might consider the following:

   - **Learn from other example scripts**: Take a look at the ``./examples/example_networks/`` directory in your PyLens folder, which contains a variety of scripts for popular neural network applications. 
   
   - **Consult the Command Reference**: For details on all available PyLens commands, including some that were not used in this tutorial, see the :doc:`Command Reference<command_reference/index>`.

