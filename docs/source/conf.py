# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Autodoc information -----------------------------------------------------
# https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-config.html


import os
import sys

sys.path.insert(0, os.path.abspath('../../easy_tpp/'))

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EasyTPP'
copyright = '2022, Machine Intelligence, Alipay'
author = 'Machine Intelligence, Alipay'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.viewcode',
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    'sphinx.ext.autosummary'
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
}
