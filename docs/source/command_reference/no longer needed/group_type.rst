.. _group_type:

Group Types in PyLens
=====================

Overview
--------

Groups in PyLens define **collections of units** with specific behaviors. Each group can be assigned a **group type**, determining how it interacts within the network.

### **Common Group Types**
- **INPUT** → Receives external data.
- **OUTPUT** → Produces final outputs.
- **HIDDEN** → Internal processing layer.
- **BIAS** → Provides a constant bias input.
- **ELMAN** → Used for Elman recurrent networks.

Setting Group Types
-------------------

You can specify a group’s type when creating it using the `add_group` function.

### **Example: Creating an Input Group**
The following code creates a **group named `"input"`** with 105 units, explicitly defining it as an **INPUT layer** with no input or output transformations:

.. code-block:: python

   simple_net.add_group(
       105,
       name="input",
       group_type="input",
       input_transforms=[],
       output_transforms=[]
   )

Modifying Group Types
---------------------

Once a group is created, its type remains **fixed** unless modified using additional PyLens functions. If you need to update or modify a group’s behavior, refer to **group modification functions**.

See Also
--------

- :meth:`add_group <your_module.add_group>`: Adds a new group with a specific type.
- :meth:`change_group_type <your_module.change_group_type>`: Modifies an existing group’s type.
- :meth:`delete_groups <your_module.delete_groups>`: Removes groups from the network.