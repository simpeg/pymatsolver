#!/usr/bin/env python
"""pymatsolver: Matrix Solvers for Python

pymatsolver is a python package for easy to use matrix solvers.

"""

import numpy as np

import os, os.path
import sys
import subprocess

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

CLASSIFIERS = [
'Development Status :: 4 - Beta',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'License :: OSI Approved :: MIT License',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Scientific/Engineering :: Mathematics',
'Topic :: Scientific/Engineering :: Physics',
'Operating System :: Microsoft :: Windows',
'Operating System :: POSIX',
'Operating System :: Unix',
'Operating System :: MacOS',
'Natural Language :: English',
]

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("find . -name \*.c -type f -delete", shell=True, executable="/bin/bash")
    subprocess.Popen("find . -name \*.so -type f -delete", shell=True, executable="/bin/bash")
    # Now do a normal clean
    sys.argv[sys.argv.index('cleanall')] = "clean"

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    cythonKwargs = dict(cmdclass={'build_ext': build_ext})
    USE_CYTHON = True
except Exception, e:
    USE_CYTHON = False
    cythonKwargs = dict()

ext = '.pyx' if USE_CYTHON else '.c'

cython_files = []
extensions = [Extension(f, [f+ext]) for f in cython_files]

if USE_CYTHON and "cleanall" not in args:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

scripts = []
if 'darwin' in sys.platform:
    subprocess.Popen("cd pymatsolver/Mumps;make build_mac", shell=True, executable="/bin/bash").wait()
else:
    subprocess.Popen("cd pymatsolver/Mumps;make build", shell=True, executable="/bin/bash").wait()
scripts += ['pymatsolver/Mumps/MumpsInterface.so','pymatsolver/Mumps/mumps_cmplx_p.f90','pymatsolver/Mumps/mumps_p.f90','pymatsolver/Mumps/mumps_interface.f90']

subprocess.Popen("cd pymatsolver/Triangle;make", shell=True, executable="/bin/bash").wait()
scripts += ['pymatsolver/Triangle/TriSolve.so','pymatsolver/Triangle/TriSolve.f']


with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name = "pymatsolver",
    version = "0.0.2",
    packages = find_packages(),
    install_requires = [
                        'numpy>=1.7',
                        'scipy>=0.13'
                       ],
    author = "Rowan Cockett",
    author_email = "rowanc1@gmail.com",
    description = "pymatsolver: Matrix Solvers for Python",
    long_description = LONG_DESCRIPTION,
    license = "MIT",
    keywords = "matrix solver",
    url = "http://simpeg.xyz/",
    download_url = "http://github.com/rowanc1/pymatsolver",
    classifiers=CLASSIFIERS,
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3 = False,
    include_dirs=[np.get_include()],
    ext_modules = extensions,
    scripts=scripts,
    **cythonKwargs
)
