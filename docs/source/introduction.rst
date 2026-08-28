Introduction to PyLens 
=====================

PyLens is a flexible neural network simulator written in Python that is based on the Lens neural network simulator (Rohde, 1999). PyLens is able to handle large, complex simulations but is also reasonably easy for novices to use. PyLens has three main design objectives:

* **Wide Range of Supported Neural Networks**. Currently, PyLens supports a wide range of neural networks that are commonly used in cognitive modeling. These include feedforward networks, simple recurrent networks, recurrent networks with backpropagation through time, continuous neural networks, and Boltzmann machines. Some of these networks, such as continuous neural networks and Boltzmann machines, are often not supported in AI-oriented neural network packages such as PyTorch and TensorFlow. Users can also freely combine components of these different networks and customize the networks they want in Python scripts.

* **Ease of Use**. PyLens includes a graphical user interface (GUI) that provides (1) visualizations of the layers, units, and weights of the neural network, (2) a command interface that users can use to configure their intended running and training settings by simply clicking buttons, and (3) a graphing component that can plot commonly used graphs, such as network training error, as well as any arbitrarily defined variables. Users can also choose to turn the GUI on for visualization or turn it off for faster computation.

* **Code Flexibility**. PyLens is primarily written in Python. The backend, including all time-sensitive calculations such as matrix calculations, is written with NumPy. The backend can also be switched to PyTorch if users want to utilize GPU computation. The graphical user interface (GUI) is written in Tkinter. Users have full access to both the backend and GUI of the whole simulator. This allows advanced users to modify the code easily and implement custom changes.

PyLens supports feedforward, simple recurrent, recurrent backpropagation through time, and continuous networks, as well as deterministic Boltzmann machines.

A PyLens script is essentially a Python script that uses PyLens modules to build neural network objects. The commonly used commands are listed in the :doc:`command reference <command_reference/index>`. Example scripts are included in ``examples/example_networks/``.

The :doc:`graphical user interface (GUI) <GUI/index>` has the following components. The Main Viewer provides a shortcut to the most frequent tasks involved in using neural networks. The Unit Viewer provides a graphical representation of the units in the network, including information about unit values and event time. The Link Viewer provides a graphical illustration of the links between the layers of the network. The Graph Viewer creates graphs of network properties, such as training error against training epoch.

PyLens also supports :doc:`parallel training <parallel>` of neural networks, so users can take advantage of the multiple cores in their CPUs. The parallel algorithm is implemented with Ray Core.
