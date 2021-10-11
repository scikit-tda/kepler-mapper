.. keplermapper documentation master file, created by
   sphinx-quickstart on Fri Dec  1 23:03:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|PyPI version| |Downloads| |Build Status|
|Codecov| |DOI zenodo| |DOI JOSS|

Kepler Mapper
=============

.. epigraph::

    Nature uses as little as possible of anything.

    -- Johannes Kepler

This is a library implementing the Mapper algorithm in Python. KeplerMapper can
be used for visualization of high-dimensional data and 3D point cloud data.
KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling
algorithms. You can find the source code on github at
`scikit-tda/kepler-mapper`_.

.. _scikit-tda/kepler-mapper: https://github.com/scikit-tda/kepler-mapper

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as
first described in the paper "Topological Methods for the Analysis of High
Dimensional Data Sets and 3D Object Recognition".


.. User's Guide
.. ------------
..
.. These pages explain what KeplerMapper is, illustrate how to use it, and discuss
.. and demonstrate applications of it.




.. toctree::
    :caption: User's Guide
    :maxdepth: 2

    theory
    started
    examples
    applications
    tutorials
    case_studies


.. API Reference
.. -------------
..
.. These pages link to documentation for specific KeplerMapper classes and functions.

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    reference/index


.. include:: citations.txt
.. include:: contributions.txt



.. |Downloads| image:: https://img.shields.io/pypi/dm/kmapper
    :target: https://pypi.python.org/pypi/kmapper/
.. |PyPI version| image:: https://badge.fury.io/py/kmapper.svg
   :target: https://badge.fury.io/py/kmapper
.. |Build Status| image:: https://travis-ci.org/scikit-tda/kepler-mapper.svg?branch=master
   :target: https://travis-ci.org/scikit-tda/kepler-mapper
.. |Codecov| image:: https://codecov.io/gh/scikit-tda/kepler-mapper/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-tda/kepler-mapper
.. |DOI JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.01315/status.svg
   :target: https://doi.org/10.21105/joss.01315
.. |DOI zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1002377.svg
   :target: https://doi.org/10.5281/zenodo.1002377
