.. _add_all_links_noise:

add_all_links_noise
===================

Applies noise to all links in the network based on a specified distribution.

Usage
-----

.. code-block:: python

   net.add_all_links_noise(noise_type="uniform", p=None, z=1)

Description
-----------

``add_all_links_noise`` introduces **noise** to all links in the network.

This function is useful for simulating weight perturbations in neural networks.

Arguments
---------

- **noise_type** (``str``, optional, default ``"uniform"``): The type of noise distribution to apply. Options:

  - ``"uniform"``: Noise is sampled from a uniform distribution.
  - ``"normal"``: Noise is sampled from a normal (Gaussian) distribution.

- **p** (``float``, optional, default ``None``): The proportion of links to which noise should be added (between 0 and 1). If ``None``, noise is applied to **all** links.

- **z** (``float``, optional, default ``1``): The standard deviation threshold for applying noise. Only links with weights within ``z`` standard deviations from the mean will receive noise.

Examples
--------

**1. Apply uniform noise to all links in the network:**

.. code-block:: python

   net.add_all_links_noise()

**2. Apply Gaussian noise to 50% of the links:**

.. code-block:: python

   net.add_all_links_noise(noise_type="normal", p=0.5)

**3. Apply noise only to links whose weights are within 2 standard deviations from the mean:**

.. code-block:: python

   net.add_all_links_noise(noise_type="uniform", z=2)

See Also
--------

- :meth:`remove_links_specific_noise <your_module.remove_links_specific_noise>`: Removes noise from specific links.
- :meth:`lesion_all_links <your_module.lesion_all_links>`: Lesions links in the network.
- :meth:`link_specific_lesion <your_module.link_specific_lesion>`: Lesions specific links between two groups.
