Installation
=============================

Supported Python Version
------------------------

PyLens supports **Python 3.14**.  


Step 1: Install Python 
------------------------

1. Visit the `Anaconda Download Page <https://docs.anaconda.com/anaconda/install/>`__ and download the installer for your operating system (Windows, macOS, or Linux).
2. Run the downloaded installer and follow the on-screen instructions.
3. Once installation is complete, open the **Anaconda Prompt** (Windows) or **Terminal** (macOS/Linux).
   
   - On Windows, look for "Anaconda Prompt" in your Start menu.

   - On macOS or Linux, open the Terminal application. Then Type:
   
   .. code-block:: bash

      conda --version

   If you see output similar to:
   
   .. code-block:: bash

      conda 4.xx.x

   that means ``conda`` is recognized, and you’re good to go!

.. tip::
    Anaconda is not necessary. It just provides a streamlined way to select different versions of Python and create virtual environments across platforms. You can also install Python using your own preferred method.

Step 2: Clone the PyLens Repository
-----------------------------------

#. If you want to clone PyLens in a specific directory on your computer, navigate to that folder. If you skip this step, the default location is typically your user's home directory.

   .. code-block:: bash

      cd ... [insert folder path here]

#. In the **Anaconda Prompt** (Windows) or **Terminal** (macOS/Linux), run the following command to clone the PyLens repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/blairarmstrong/PyLensPublic

#. Once cloning is complete, navigate into the newly created ``PyLens`` folder:

   .. code-block:: bash

      cd PyLensPublic

.. tip::
   1. Alternatively, download the repository ZIP file from GitHub and unzip it. Then, cd into the unzipped ``PyLensPublic`` directory.
   2. If you don’t have Git installed, you can `download Git <https://git-scm.com/downloads>`__ and install it.  

Step 3: Create a Virtual Environment
---------------------------------------------------

Creating a virtual environment isolates the packages required for PyLens, preventing conflicts with other projects or system-wide packages.

1. Make sure you are still in your **Anaconda Prompt** (Windows) or **Terminal** (macOS/Linux) where ``conda`` is available.
2. Run the following command to create a new environment named (for example) ``pylens_env`` with Python 3.11:

   .. code-block:: bash

      conda create --name pylens_env python=3.14

   When asked to proceed, type ``y`` and press **Enter**.

3. Activate the newly created environment:

   .. code-block:: bash

      conda activate pylens_env

4. After activation, you should see ``(pylens_env)`` at the beginning of your command prompt. This indicates you are now working inside the ``pylens_env`` environment.

.. tip::
    You can also initialize a virtual environment with your favoriate methods.

Step 4: Install PyLens and Dependencies
---------------------------------------

1. Ensure you are in the ``PyLensPublic`` directory and have your conda environment (``pylens_env``) active.
2. Install PyLens using ``pip`` (which comes with Python):

   .. code-block:: bash

      pip install -e .

3. If you plan to develop PyLens, also install the extra development dependencies:

   .. code-block:: bash

      pip install -e .[dev]

.. note::
   - Always remember to activate your ``pylens_env`` environment (``conda activate pylens_env``) before working on PyLens or installing any new packages.

Step 5: Verify Installation
---------------------------

#. To check if PyLens installed correctly, run:

   .. code-block:: bash

      pip show pylens

#. If PyLens is installed, this command will display details like its version, location, and dependencies.

Troubleshooting and Tips
------------------------

- **Environment Activation**: If you open a new terminal or Anaconda Prompt, you must re-activate your environment each time:

  .. code-block:: bash

     conda activate pylens_env

- **Python Version**: Verify your Python version is 3.14 by running:

  .. code-block:: bash

     python --version

  If it’s not 3.14 make sure you have activated the correct conda environment or installed Python 3.14 correctly.


Congratulations!
----------------

You have successfully installed and set up PyLens in a dedicated conda environment with Python 3.14
