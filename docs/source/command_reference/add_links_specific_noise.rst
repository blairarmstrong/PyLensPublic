.. _add_links_specific_noise:

add_links_specific_noise
========================

Applies noise to specific links between two groups in the network.

Usage
-----

.. code-block:: python

   net.add_links_specific_noise(
       outgoing_group, 
       incoming_group, 
       noise_type="uniform", 
       p=None, 
       z=1
   )

Description
-----------

``add_links_specific_noise`` introduces **noise** to links connecting a specific pair of groups in the network.

This function is useful for simulating targeted weight perturbations in neural networks.

Arguments
---------

- **outgoing_group** (``str``): The name of the group that sends data forward.

- **incoming_group** (``str``): The name of the group receiving the links.

- **noise_type** (``str``, optional, default ``"uniform"``): The type of noise distribution to apply. Options:

  - ``"uniform"``: Noise is sampled from a uniform distribution.
  - ``"normal"``: Noise is sampled from a normal (Gaussian) distribution.

- **p** (``float``, optional, default ``None``): The proportion of links to which noise should be added (between 0 and 1). If ``None``, noise is applied to **all** links between the specified groups.

- **z** (``float``, optional, default ``1``): The standard deviation threshold for applying noise. Only links with weights within ``z`` standard deviations from the mean will receive noise.

Examples
--------

**1. Apply uniform noise to all links between "hidden1" and "output":**

.. code-block:: python

   net.add_links_specific_noise("hidden1", "output")

**2. Apply Gaussian noise to 50% of the links between "input" and "hidden":**

.. code-block:: python

   net.add_links_specific_noise("input", "hidden", noise_type="normal", p=0.5)

**3. Apply noise only to links between "hidden1" and "hidden2" whose weights are within 2 standard deviations from the mean:**

.. code-block:: python

   net.add_links_specific_noise("hidden1", "hidden2", noise_type="uniform", z=2)

See Also
--------

- :meth:`add_all_links_noise <your_module.add_all_links_noise>`: Applies noise to all links in the network.
- :meth:`remove_links_specific_noise <your_module.remove_links_specific_noise>`: Removes noise from specific links.
- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions all links in the network.
