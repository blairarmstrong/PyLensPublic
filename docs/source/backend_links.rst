Backend: Links
================

Links are object representing weights in a neural network. There are 3 types of link: Random, One-To-One and Full.

- **Random link** represents lesioned link where p% of connection are dropped permenantly.
- **One-to-One** represents one to one connection between two groups, i.e each units from the previous group will contribute to the corresponding units in the next group.
- **Full Link** represent full connectivity between two groups, i.e every units from the previous group will contribute to every units in the next group

Usage Example
-------------
.. code-block:: python

    Network.connect_groups("first_layer", "second_layer", link_type="inhibitory", proj_type="random", drop_rate=0.7)

In this case, the sparse matrix will be used as masks.

In contrast:

.. code-block:: python

    Network.connect_groups("first_layer", "second_layer", link_type="inhibitory", proj_type="random", drop_rate=0.2)

In this case, the dense matrix will be used.

Programmer's guide on sparse vs dense mask choice
-------------------------------------------------

**Dot product**

For dot product between n by n weight matrix by a size n vector.

Sparse matrix has a timing advantage when over 75% of the cells are unfiled.

The time performance: Scipy is y times faster than numpy can be modelled by y = -1/(4(p-1)) as n increases, where p is the drop rate.

**Elementwise matrix multiplication**

For dot product between n by n weight matrix by a size n by n matrix.

Sparse matrix has a timing advantage when over 60% of the cells are unfilled.

The time performance: Scipy is y times faster than numpy can be modelled by y = -3/(8(p-1)) as n increases, where p is the drop rate.

Currently the code is doing the switch operation automatically, such that when the matrix will be converted to sparse type if the sparsity is over certain threshold as determined above. Otherwise it will use numpy dense array.
