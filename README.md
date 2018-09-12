[![PyPI version](https://badge.fury.io/py/kmapper.svg)](https://badge.fury.io/py/kmapper)
[![Build Status](https://travis-ci.org/MLWave/kepler-mapper.svg?branch=master)](https://travis-ci.org/MLWave/kepler-mapper)
[![Codecov](https://codecov.io/gh/mlwave/kepler-mapper/branch/master/graph/badge.svg)](https://codecov.io/gh/mlwave/kepler-mapper)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1054444.svg)](https://doi.org/10.5281/zenodo.1054444)


# KeplerMapper <img align="right" width="40" height="40" src="http://i.imgur.com/axOG6GJ.jpg">

> Nature uses as little as possible of anything. - Johannes Kepler

This is a Python implementation of the TDA Mapper algorithm  for visualization of high-dimensional data. For complete documentation, see [https://MLWave.github.io/kepler-mapper](https://MLWave.github.io/kepler-mapper).



KeplerMapper employs approaches based on the Mapper algorithm (Singh et al.) as first described in the paper "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition".

KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling algorithms.


## Install

### Dependencies

KeplerMapper requires:

  - Python (>= 2.7 or >= 3.3)
  - NumPy
  - Scikit-learn
  - Python-Igraph
  - Plotly 
  - Ipywidgets
  
Running some of the examples requires:

  - matplotlib

Visualizations load external resources:

  - Roboto Webfont (Google)
  - D3.js (Mike Bostock)


### Installation

Install KeplerMapper with pip:

```
pip install kmapper
```

To install from source:
```
git clone https://github.com/MLWave/kepler-mapper
cd kepler-mapper
pip install -e .
```


### For   Plotly visualization and interaction with kmapper graph
via ipywidgets  you need:

`notebook` version >=5.3 and `ipywidgets` >=7.2

Install both either with pip:

```
pip install "notebook>=5.3" "ipywidgets>=7.2"
```
or with conda:

```
conda install "notebook>=5.3" "ipywidgets>=7.2"
```

Install Plotly with pip:

```
pip install plotly
```

or with conda:
```
conda install -c plotly plotly
```

Install graph library `python-igraph` with pip:
```
pip install python-igraph
```
Windows OS users can download an adequate installer
from  [https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph](https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph)

and install it via pip:
```
pip installer_name.whl
```

To save the Plotly graph as a raster (png, jpg, webp) or vector format (pdf, svg)
install plotly-orca, and the psutil Python package:

```
conda install -c plotly plotly-orca psutil
```
## Usage

KeplerMapper adopts the scikit-learn API as much as possible, so it should feel very familiar to anyone who has used these libraries.

### Python code
```python
# Import the class
import kmapper as km

# Some sample data
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1]) # X-Y axis

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, nr_cubes=10)

# Visualize it via d3.js
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")

#or as a Plotly plot:

from kmapper.plotlyviz import *
import plotly.graph_objs as go

kmgraph,  mapper_summary, n_color_distribution = get_mapper_graph(simplicial_complex, 
                                                                  color_function=None)
pl_data = plotly_graph(kmgraph, 
                       graph_layout='kk')

layout = plot_layout(title='Mapper Graph for circles',
                     bgcolor='rgba(240, 240, 240, 0.95)')
fw = go.FigureWidget(data=pl_data, layout=layout)

fw                
```

## Disclaimer

Standard MIT disclaimer applies, see `DISCLAIMER.md` for full text. Development status is Alpha.

## Cite

Nathaniel Saul, & Hendrik Jacob van Veen. (2017, November 17). MLWave/kepler-mapper: 186f (Version 1.0.1). Zenodo. http://doi.org/10.5281/zenodo.1054444
