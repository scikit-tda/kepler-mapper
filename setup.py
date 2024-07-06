#!/usr/bin/env python

from setuptools import setup

import re

VERSIONFILE = "kmapper/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open("README.md") as f:
    long_description = f.read()

setup(
    name="kmapper",
    version=verstr,
    description="Python implementation of Mapper algorithm for Topological Data Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HJ van Veen, Nathaniel Saul, David Eargle, Sam Mangham",
    author_email="info@mlwave.com, nat@saulgill.com",
    url="http://kepler-mapper.scikit-tda.org",
    license="MIT",
    packages=["kmapper"],
    include_package_data=True,
    extras_require={
        "testing": [  # `pip install -e ".[testing]"``
            "pytest",  # ~=6.2.5",
            "networkx",  # ~=2.5.1",
            "matplotlib",  # ~=3.3.4",
            "igraph",
            "plotly",  # ~=5.3.1",
            "ipywidgets",  # ~=7.6.5",
        ],
        "docs": [  # `pip install -e ".[docs]"``
            "sktda_docs_config",  # latest
            "sphinx",  # ~=4.2.0",
            "pandas",  # ~=1.1.5",
            "sphinx-gallery",  # ~=0.10.0",
            # for building docs for plotlyviz stuff
            "networkx",  # ~=2.5.1",
            "matplotlib",  # ~=3.3.4",
            "igraph",
            "plotly",  # ~=5.3.1",
            "ipykernel",
            "ipywidgets",  # ~=7.6.5",
            "ipython",  # ~=7.16.1",
            "nbsphinx",  # ~=0.8.7",
            # required for building some jupyter notebooks.
            # uncomment if rebuilding the notebooks.
            ## Plotly-Demo.ipynb
            # "cmocean~=2.0",
            # "kaleido~=0.2.1",
            ## Confidence-Graphs.ipynb
            # "tensorflow~=2.2.0",
            # "pillow",
            # "xgboost",
            # "scikit-image"
        ],
    },
    install_requires=["scikit-learn", "numpy", "scipy", "Jinja2"],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="mapper, topology data analysis, algebraic topology, unsupervised learning",
)
