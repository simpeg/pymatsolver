#!/usr/bin/env python
"""pymatsolver: Matrix Solvers for Python

pymatsolver is a python package for easy to use matrix solvers.

"""

from setuptools import setup, find_packages

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

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name="pymatsolver",
    version="0.2.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        'numpy>=1.7',
        'scipy>=1.8',
    ],
    author="SimPEG Developers",
    author_email="rowanc1@gmail.com",
    description="pymatsolver: Matrix Solvers for Python",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="matrix solver",
    url="http://simpeg.xyz/",
    download_url="http://github.com/simpeg/pymatsolver",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False
)
