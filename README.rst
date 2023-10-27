Exodus Helper Package
=====================

|docs| |tests| |lint|

Description/Purpose Statement
-----------------------------
This package provides native Python tools to handle ExodusII files through the NetCDF API.

Installation
------------

- Clone the |project| repo to your local disk and navigate to the created folder:

::

   $ git clone git@github.com:sandialabs/exodus_helper.git
   $ cd exodus_helper

- (Users): To install |project| locally

::

    $ python -m pip install .

- (Developer) To install |project| so that local edits to the soure code are immediately available system-wide:

::

   $ python -m pip install -e .

- To include extras, e.g. for testing (see tool.poetry.extras in pyproject.toml for available options), run:

::

   $ python -m pip install ".[testing]"

Getting Started
---------------

- `Documentation <https://exodus-helper.readthedocs.io/en/latest/index.html/>`_
- Release History (link pending)
- Tutorial Notebook (link pending)
- Contribution Guidelines (link pending)


.. |project| replace:: exodus_helper

.. |docs| image:: https://readthedocs.org/projects/exodus-helper/badge/?version=latest&style=flat
    :target: https://exodus-helper.readthedocs.io/en/latest/index.html
    :alt: docs

.. |tests| image:: https://github.com/sandialabs/exodus_helper/actions/workflows/pytest.yml/badge.svg
    :target: https://github.com/sandialabs/exodus_helper/actions/workflows/pytest.yml
    :alt: tests

.. |lint| image:: https://github.com/sandialabs/exodus_helper/actions/workflows/pylint.yml/badge.svg
    :target: https://github.com/sandialabs/exodus_helper/actions/workflows/pylint.yml
    :alt: lint
