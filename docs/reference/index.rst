
API Reference
---------------

.. currentmodule:: kmapper

Kepler Mapper
===============

The ``KeplerMapper`` class provides three main functions: ``fit_transform, map, and visualize``. 
These provide the bulk of the functionality for constructing a Mapper graph. 
Cover schemes provide a customizable way of defining a cover for your lens. 
Nerves determine the method that constructs the nodes and edges (or higher order connections).


.. autosummary::
    :toctree: stubs
    :nosignatures:

    KeplerMapper
    Cover
    GraphNerve


Visuals
==========

There are many ways to visualize your constructed Mapper. The original and most popular method is to use the ``KeplerMapper.visualize`` function to construct an ``.html`` file. You can then open this file in your browser and explore the result. If you use Jupyter Notebooks, you can visualize the ``.html`` file directly in a notebook using the ``kmapper.jupyter.display`` function. We also provide extensive functionality for constructing Plotly graphs and dashboards using the ``kmapper.plotlyviz`` module.  To learn more about the Plotlyviz functionality, see the `Jupyter demo`_ or `Plotly demo`_. 



.. autosummary::
    :toctree: stubs
    :nosignatures:

    jupyter
    plotlyviz.plotlyviz
    plotlyviz.mpl_to_plotly
    draw_matplotlib
    visuals.colorscale_from_matplotlib_cmap


Adapters
==========

We provide a basic adapter to convert a KeplerMapper graph to `networkx.Graph` format. 

.. note::

  If you would like adapters to other popular formats, please let us know.


.. autosummary::
    :toctree: stubs
    :nosignatures:

    kmapper.adapter




.. _Visuals documentation page: visuals.html

.. _Jupyter demo: ../notebooks/KeplerMapper-usage-in-Jupyter-Notebook.ipynb

.. _Plotly demo: ../notebooks/Plotly-Demo.ipynb