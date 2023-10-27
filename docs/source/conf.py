"""Configuration file for the Sphinx documentation builder.
For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html"""

import importlib.metadata
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../examples'))


def get_version():
    return importlib.metadata.version('exodus-helper')

# Project information ------------------------------------------------------- #
# www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'exodus_helper'
copyright = '2023, Coleman Alleman'
author = 'Coleman Alleman'
release = get_version()
tmp = release.split('.')
version = str(f'{tmp[0]}.{tmp[1]}')

# General configuration ----------------------------------------------------- #
# www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode']

autodoc_member_order = 'groupwise'
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

# Options for HTML output --------------------------------------------------- #
# www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# Restrict autodoc to include only information about documented functions
autodoc_default_flags = ['members']
