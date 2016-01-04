pymatsolver
***********

.. image:: https://img.shields.io/pypi/v/pymatsolver.svg
    :target: https://crate.io/packages/pymatsolver/
    :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/dm/pymatsolver.svg
    :target: https://crate.io/packages/pymatsolver/
    :alt: Number of PyPI downloads

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/rowanc1/pymatsolver/blob/master/LICENSE
    :alt: MIT license.

.. image:: https://img.shields.io/travis/rowanc1/pymatsolver.svg
    :target: https://travis-ci.org/rowanc1/pymatsolver
    :alt: Travis CI build status

.. image:: https://img.shields.io/coveralls/rowanc1/pymatsolver.svg
    :target: https://coveralls.io/r/rowanc1/pymatsolver?branch=master
    :alt: Coverage status


A sparse matrix solver for python.

Solving Ax = b should be as easy as:

.. code-block:: python

    Ainv = Solver(A)
    x = Ainv * b

In pymatsolver I provide a number of wrappers to existing numerical packages. Nothing fancy here.


Installing
==========

I have not been able to get the pip install to work because of multiple dependencies on fortran libraries.
However, the linux and mac installs are relatively easy. Note that you must have mumps pre-installed,
currently I have only got this working for the sequential version, so when you are installing,
you will need to point to that one. You can also look at the `.travis.yml` file for how to get it working on TravisCI.

Linux
-----

From a clean install on Ubuntu:

.. code-block:: bash

    apt-get update
    apt-get -y install gcc gfortran git libopenmpi-dev libmumps-seq-dev libblas-dev liblapack-dev

    # Install all the python you need!
    wget http://repo.continuum.io/miniconda/Miniconda-3.8.3-Linux-x86_64.sh -O miniconda.sh;
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=/root/anaconda/bin:/root/miniconda/bin:$PATH
    conda update --yes conda
    conda install --yes numpy scipy matplotlib cython ipython nose

    git clone https://github.com/rowanc1/pymatsolver.git
    cd pymatsolver
    python setup.py install

Mac
---

This assumes that you have Brew and some python installed (numpy, scipy):

.. code-block:: bash

    brew install mumps --with-scotch5 --without-mpi

    git clone https://github.com/rowanc1/pymatsolver.git
    cd pymatsolver
    python setup.py install


Code:
https://github.com/rowanc1/pymatsolver


Tests:
https://travis-ci.org/rowanc1/pymatsolver


Bugs & Issues:
https://github.com/rowanc1/pymatsolver/issues
