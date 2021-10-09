.. keplermapper documentation master file, created by
   sphinx-quickstart on Fri Dec  1 23:03:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|PyPI version| |Downloads| |Build Status|
|Codecov| |DOI zenodo| |DOI JOSS|

Kepler Mapper
=============


    Nature uses as little as possible of anything. - Johannes Kepler

This is a library implementing the Mapper algorithm in Python. KeplerMapper can
be used for visualization of high-dimensional data and 3D point cloud data.
KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling
algorithms. You can find the source code on github at
`scikit-tda/kepler-mapper`_.

.. _scikit-tda/kepler-mapper: https://github.com/scikit-tda/kepler-mapper

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as
first described in the paper "Topological Methods for the Analysis of High
Dimensional Data Sets and 3D Object Recognition".


User's Guide
------------

These pages explain what KeplerMapper is, illustrate how to use it, and discuss
and demonstrate applications of it.

.. toctree::
    :maxdepth: 2

    theory
    started
    examples
    applications
    tutorials
    case_studies


API Reference
-------------

These pages link to documentation for specific KeplerMapper classes and functions.

.. toctree::
    :maxdepth: 3

    reference/index


Citations
---------

To credit KeplerMapper in your work, please cite both the `JOSS paper`_
and the `Zenodo archive`_. The former provides a high level description
of the package, and the latter points to a permanent record of all KeplerMapper versions
(we encourage you to cite the specific version you used).

.. _JOSS paper: https://doi.org/10.21105/joss.01315
.. _Zenodo archive: https://doi.org/10.5281/zenodo.1002377

Example citations (for KeplerMapper 1.4.1):

    van Veen et al., (2019). Kepler Mapper: A flexible Python implementation of the Mapper algorithm.
    Journal of Open Source Software, 4(42), 1315, https://doi.org/10.21105/joss.01315

    Hendrik Jacob van Veen, Nathaniel Saul, David Eargle, & Sam W. Mangham.
    (2019, October 14). Kepler Mapper: A flexible Python implementation of the Mapper algorithm (Version 1.4.1).
    Zenodo. http://doi.org/10.5281/zenodo.4077395

Bibtex entry for JOSS article:

::

  @article{KeplerMapper_JOSS,
      doi           = {10.21105/joss.01315},
      url           = {https://doi.org/10.21105/joss.01315},
      year          = {2019},
      publisher     = {The Open Journal},
      volume        = {4},
      number        = {42},
      pages         = {1315},
      author        = {Hendrik Jacob van Veen and Nathaniel Saul and David Eargle and Sam W. Mangham},
      title         = {Kepler Mapper: A flexible Python implementation of the Mapper algorithm.},
      journal       = {Journal of Open Source Software}
      }

Bibtex entry for the Zenodo archive, version 1.4.1:

::

    @software{KeplerMapper_v1.4.1-Zenodo,
        author       = {Hendrik Jacob van Veen and
                        Nathaniel Saul and
                        Eargle, David and
                        Sam W. Mangham},
        title        = {{Kepler Mapper: A flexible Python implementation of
                         the Mapper algorithm}},
        month        = oct,
        year         = 2020,
        publisher    = {Zenodo},
        version      = {1.4.1},
        doi          = {10.5281/zenodo.4077395},
        url          = {https://doi.org/10.5281/zenodo.4077395}
    }



Contributions
-------------

We welcome contributions of all shapes and sizes. There are lots of
opportunities for potential projects, so please get in touch if you would like
to help out. Everything from an implementation of your favorite distance,
notebooks, examples, and documentation are all equally valuable so please don’t
feel you can’t contribute.

See `the contribution guideline page in the source code repository`__ for more details.

__ https://github.com/scikit-tda/kepler-mapper/blob/master/CONTRIBUTING.md


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
