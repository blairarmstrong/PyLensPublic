Jupyter Integration (WIP)
==========================================

Set-up
--------------
In order to integrate Jupyter Notebook with PyLens and the GUI, make sure all the PyLens requirements are installed. This can be ensured by running inside the virtual environment of the project (Need to update).
All the requirements must be downloaded just to make sure the virtual environment is fully configured with the following: :code:`pip install -r requirements.txt`. Lastly, it is essential to correctly configure the package directory which can be down through: :code:`python setup.py install`

Usage
--------------
Once all the set-up is complete, access the root directory of PyLens either using *terminal* or the *command prompt* and launch the Jupyter Notebook with the following command: :code:`jupyter notebook`. When creating a new notebook, make sure to always run :code:`%gui tk` before running any code for successful integration of the GUI.

Example using XOR
------------------------
An XOR example can be run with the following steps.
First begin by running :code:`%gui tk` on the first line to integrate the GUI in the Jupyter Notebook kernel.
Import the necessary modules with the following lines of code

.. code-block:: python

   from src.backend.network import Network
   from src.gui.unit_viewer_tk import FrameExamplesProgram
   from src.simulator import Simulator

Then, build the simulator and the network and load the appropriate example set. Include the final line of code to visualize the Network with the Unit Viewer.

.. code-block:: python

   s_xor = Simulator("simulator for xor")
   mynet = Network("mynet")  # default time intervals (1) ->
   mynet.add_group("first", 2, "input", [], [])
   mynet.add_group("second", 2, None, ["dot"], ["sigmoid"])
   mynet.add_group("third", 1, "output", ["dot"], ["sigmoid"])
   mynet.connect_groups("first", "second", "uniform")
   mynet.connect_groups("second", "third", "uniform")
   mynet.load_example_set("XOR",  False,  "examples/example_files/xor_sparse.ex")
   s_xor.use_gui(mynet)

Once the training of the network is complete, the unit viewer should appear like the following.

.. figure:: ../images/unit_viewer.jpg
   :alt: the unit viewer with a XOR example loaded
   :width: 600
   :align: center

   The unit viewer with the XOR example loaded
