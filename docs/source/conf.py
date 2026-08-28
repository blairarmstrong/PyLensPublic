# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------

project = 'PyLens'
copyright = '2020, CAP Lab, University of Toronto'
author = 'CAP Lab'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx_design',
    "sphinx.ext.napoleon",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

python_maximum_signature_line_length = 60

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_theme_options = {
    'show_powered_by': False,
    'fixed_sidebar': True,
    'sidebar_width': '280px',
    'globaltoc_depth': 4,
    'globaltoc_collapse': False,
}

html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = False

def setup(app):
    app.add_css_file('custom.css')

# store definitions for PyLens commands so they can be reused in different places in the command reference
rst_prolog = """
.. |create_net| replace:: Create a new network instance based on the specified type
.. |add_net| replace:: Adds a new network instance to the simulator
.. |use_net| replace:: Set the specified network as the active network
.. |delete_net| replace:: Remove a network instance from the simulator
.. |store_network| replace:: Save the current network state to a file in either pickle or JSON format
.. |load_network| replace:: Load a previously saved network from a pickle or JSON file
.. |set_time| replace:: Set the network's time parameters 
.. |add_group| replace:: Add a group to an existing network
.. |delete_group| replace:: Delete a group from an existing network
.. |order_groups| replace:: Set the order in which groups are updated
.. |get_group_type| replace:: Print information about a group’s type, input type, and output type
.. |change_group_type| replace:: Add new input and/or output transforms to a group
.. |copy_unit_values| replace:: Copy unit values from one group to another group or within the same group
.. |print_unit_values| replace:: Print basic unit values for a list of groups into a file
.. |reset_unit_values| replace:: Reset values of a specified field for all units in the group
.. |polarity| replace:: Compute the average polarization of unit outputs in a group
.. |connect_groups| replace:: Create links with a specified pattern between groups
.. |connect_group_to_unit| replace:: Connect all units in one group to specific units in another group
.. |connect_units| replace:: Create links between specific units
.. |elman_connect| replace:: Initialize weights for an Elman connection between two groups
.. |copy_connect| replace:: Copy values from a specific field of another group
.. |disconnect_groups| replace:: Delete links of a specified type between two groups
.. |disconnect_group_units| replace:: Delete links from a group to one or more units
.. |disconnect_units| replace:: Delete links of a specified type between two units
.. |delete_links| replace:: Delete all links of a specified type
.. |delete_group_inputs| replace:: Delete incoming links to a group
.. |delete_group_outputs| replace:: Delete outgoing links from a group
.. |delete_unit_inputs| replace:: Delete incoming links to a unit
.. |delete_unit_outputs| replace:: Delete outgoing links from a unit

.. |train| replace:: Train the network using a specified algorithm
.. |set_update_method| replace:: Set the weight update method for training
.. |update_weights| replace:: Update the weights of the links between groups
.. |test| replace:: Evaluate the network on the testing set
.. |reset_network| replace:: Reset the network to its initial state
.. |reset_derivs| replace:: Reset the derivative values for all groups in the network
.. |save_stats| replace:: Save the training statistics to a file

"""
