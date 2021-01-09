.. keplermapper documentation master file, created by
   sphinx-quickstart on Fri Dec  1 23:03:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|DOI| |PyPI version| |Downloads| 
|Build Status| |Codecov| 

Kepler Mapper
----------------


    Nature uses as little as possible of anything. - Johannes Kepler

This is a library implementing the Mapper algorithm in Python. KeplerMapper can be used for visualization of high-dimensional data and 3D point cloud data. KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling algorithms. You can find the source code on github at 
`scikit-tda/kepler-mapper <https://github.com/scikit-tda/kepler-mapper>`_.

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as first described in the paper "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition".

Setup
=======================

Install KeplerMapper with pip:

::

    pip install kmapper



Citations
=======================

Hendrik Jacob van Veen, and Nathaniel Saul. (2017, November 17).
MLWave/kepler-mapper: 186f (Version 1.0.1). Zenodo.
http://doi.org/10.5281/zenodo.1054444



Bibtex entry:


::

    @MISC {KeplerMapper2019,
        author       = "Hendrik Jacob van Veen and Nathaniel Saul",
        title        = "KeplerMapper",
        howpublished = "http://doi.org/10.5281/zenodo.1054444",
        month        = "Jan",
        year         = "2019"
    }



Contributions
=======================

We welcome contributions of all shapes and sizes. There are lots of opportunities for potential projects, so please get in touch if you would like to help out. Everything from an implementation of your favorite distance, notebooks, examples, and documentation are all equally valuable so please don’t feel you can’t contribute.

To contribute please fork the project make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: User Guide

    theory
    started
    applications
    reference/index

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Tutorials

    notebooks/Adapters
    notebooks/Plotly-Demo
    notebooks/Cancer-demo
    notebooks/KeplerMapper-usage-in-Jupyter-Notebook

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Advanced Case Studies

    notebooks/KeplerMapper-Newsgroup20-Pipeline
    notebooks/TOR-XGB-TDA
    notebooks/Confidence-Graphs
    notebooks/self-guessing


.. |Downloads| image:: https://pypip.in/download/kmapper/badge.svg
    :target: https://pypi.python.org/pypi/kmapper/
.. |PyPI version| image:: https://badge.fury.io/py/kmapper.svg
   :target: https://badge.fury.io/py/kmapper
.. |Build Status| image:: https://travis-ci.org/scikit-tda/kepler-mapper.svg?branch=master
   :target: https://travis-ci.org/scikit-tda/kepler-mapper
.. |Codecov| image:: https://codecov.io/gh/scikit-tda/kepler-mapper/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-tda/kepler-mapper
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1054444.svg
   :target: https://doi.org/10.5281/zenodo.1054444
