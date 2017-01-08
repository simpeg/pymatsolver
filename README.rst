pymatsolver
***********

.. image:: https://img.shields.io/pypi/v/pymatsolver.svg
    :target: https://crate.io/packages/pymatsolver/
    :alt: Latest PyPI version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/rowanc1/pymatsolver/blob/master/LICENSE
    :alt: MIT license.

.. image:: https://img.shields.io/travis/rowanc1/pymatsolver.svg
    :target: https://travis-ci.org/rowanc1/pymatsolver
    :alt: Travis CI build status

.. image:: https://codecov.io/gh/rowanc1/pymatsolver/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/rowanc1/pymatsolver
    :alt: Coverage status


A (sparse) matrix solver for python.

Solving Ax = b should be as easy as:

.. code-block:: python

    Ainv = Solver(A)
    x = Ainv * b

In pymatsolver I provide a number of wrappers to existing numerical packages. Nothing fancy here.

Solvers Available
=================

All solvers work with :code:`scipy.sparse` matricies, and a single or multiple right hand sides using :code:`numpy`:

* L/U Triangular Solves
* Wrapping of SciPy matrix solvers (direct and indirect)
* Pardiso solvers now that MKL comes with conda!


Code:
https://github.com/rowanc1/pymatsolver


Tests:
https://travis-ci.org/rowanc1/pymatsolver


Bugs & Issues:
https://github.com/rowanc1/pymatsolver/issues

License: MIT
