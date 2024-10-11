pymatsolver
***********

.. image:: https://img.shields.io/pypi/v/pymatsolver.svg
    :target: https://pypi.python.org/pypi/pymatsolver
    :alt: Latest PyPI version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/simpeg/pymatsolver/blob/master/LICENSE
    :alt: MIT license.

.. image:: https://codecov.io/gh/simpeg/pymatsolver/branch/main/graph/badge.svg?token=8uQoxzxf3r
    :target: https://codecov.io/gh/simpeg/pymatsolver
    :alt: Coverage status


A (sparse) matrix solver for python.

Solving Ax = b should be as easy as:

.. code-block:: python

    Ainv = Solver(A)
    x = Ainv * b

In pymatsolver we provide a number of wrappers to existing numerical packages. Nothing fancy here.

Solvers Available
=================

All solvers work with :code:`scipy.sparse` matricies, and a single or multiple right hand sides using :code:`numpy`:

* L/U Triangular Solves
* Wrapping of SciPy matrix solvers (direct and indirect)
* Pardiso solvers
* Mumps solvers


Installing Solvers
==================
Often, there are faster solvers available for your system than the default scipy factorizations available.
pymatsolver provides a consistent interface to both MKL's ``Pardiso`` routines and the ``MUMPS`` solver package. To
make use of these we use intermediate wrappers for the libraries that must be installed separately.

Pardiso
-------
The Pardiso interface is recommended for Intel processor based systems. The interface is enabled by
the ``pydiso`` python package, which can be installed through conda-forge as:

.. code::

    conda install -c conda-forge pydiso

Mumps
-----
Mumps is available for all platforms. The mumps interface is enabled by installing the ``python-mumps``
wrapper package. This can easily be installed through conda-forge with:

.. code::

    conda install -c conda-forge python-mumps



Code:
https://github.com/simpeg/pymatsolver


Tests:
https://github.com/simpeg/pymatsolver/actions


Bugs & Issues:
https://github.com/simpeg/pymatsolver/issues

License: MIT
