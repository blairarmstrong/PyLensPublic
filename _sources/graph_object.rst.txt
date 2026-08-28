.. _graph_object:

graphObject
===========

Opens a live-updating graph to visualize the value of a network object or variable.

Usage
-----

.. code-block:: python

   graph = graphObject(parent, network, plot_variable)

Description
-----------

``graphObject`` creates a **real-time graph** that tracks the value of a specified **network object or variable** during simulation. This provides a **visual representation of training progress**, unit activations, or error trends.

If no variable is specified, **error** is used as the default.

### **Tracked Objects**
The ``plot_variable`` argument defines what the graph tracks. Examples include:
- **"error"** → Monitors the network’s training error.
- **"group.hidden.output"** → Tracks the output of all units in the hidden layer.
- **"unit.2.input"** → Displays input values for unit 2.
- **"group(3).unit(2).output"** → Plots the activation of a specific unit.

### **Update Frequency**
The graph updates at a fixed interval (`interval=100ms` by default).  
Possible update triggers include:
- **TICKS** → Updates after each tick.
- **EVENTS** → Updates based on defined events.
- **EXAMPLES** → Updates after each training example.
- **WEIGHT_UPDATES** → Updates after weight changes.
- **PROGRESS_REPORTS** *(default)* → Updates periodically during training.
- **TRAINING_AND_TESTING** → Updates at both training and test phases.
- **USER_SIGNALS** → Updates when triggered manually.

Arguments
---------

- **parent** (``Tk``): The Tkinter window or parent widget in which the graph is embedded.

- **network** (``Network``): The active network whose values will be tracked.

- **plot_variable** (``str``): The **network parameter** (e.g., ``"error"``, ``"group.hidden.output"``, ``"unit.2.input"``) to be visualized.

Features
--------

- **Automatic Plot Updates** → The graph refreshes every ``100ms`` (default) or based on an update trigger.
- **Supports Multiple Traces** → Tracks multiple variables at once.
- **Interactive Toolbar** → Allows zooming, panning, and saving figures.
- **Customizable Axes** → Dynamically adjusts graph limits as new data arrives.
- **Graceful Shutdown** → Ensures proper cleanup when the graph window is closed.

Examples
--------

**1. Graph the network error over time:**

.. code-block:: python

   graph = graphObject(simulator, my_network, "error")

**2. Visualize the output of units 0, 1, and 2 every tick:**

.. code-block:: python

   graph = graphObject(simulator, my_network, "out:0.output out:1.output out:2.output")

See Also
--------

- :meth:`train <your_module.train>`: Trains the network and generates data for visualization.
- :meth:`test <your_module.test>`: Evaluates the network and updates graph traces.
- :meth:`reset_network <your_module.reset_network>`: Resets the simulation and clears stored values.
