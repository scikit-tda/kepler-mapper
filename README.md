[![PyPI version](https://badge.fury.io/py/kmapper.svg)](https://badge.fury.io/py/kmapper)
[![Downloads](https://pypip.in/download/kmapper/badge.svg)](https://pypi.python.org/pypi/kmapper/)
[![Build Status](https://travis-ci.org/scikit-tda/kepler-mapper.svg?branch=master)](https://travis-ci.org/scikit-tda/kepler-mapper)
[![Codecov](https://codecov.io/gh/scikit-tda/kepler-mapper/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-tda/kepler-mapper)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.01315/status.svg)](https://doi.org/10.21105/joss.01315)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1002377.svg)](https://doi.org/10.5281/zenodo.1002377)

# KeplerMapper <img align="right" width="40" height="40" src="http://i.imgur.com/axOG6GJ.jpg">

> Nature uses as little as possible of anything. - Johannes Kepler

This is a Python implementation of the TDA Mapper algorithm  for visualization of high-dimensional data. For complete documentation, see [https://kepler-mapper.scikit-tda.org](https://kepler-mapper.scikit-tda.org).

KeplerMapper employs approaches based on the Mapper algorithm (Singh et al.) as first described in the paper "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition".

KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling algorithms.


## Install

### Dependencies

KeplerMapper requires:

  - Python (>= 2.7 or >= 3.3)
  - NumPy
  - Scikit-learn

Using the plotly visualizations requires a few extra libraries:

  - Python-Igraph
  - Plotly
  - Ipywidgets

Additionally, running some of the examples requires:

  - matplotlib
  - umap-learn


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
graph = mapper.map(projected_data, data, cover=km.Cover(n_cubes=10))

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
```

## Disclaimer

Standard MIT disclaimer applies, see `DISCLAIMER.md` for full text. Development status is Alpha.

## How to cite

To credit KeplerMapper in your work, please cite both the [JOSS paper](https://doi.org/10.21105/joss.01315) and the [Zenodo archive](https://doi.org/10.5281/zenodo.1002377). The former provides a high level description of the package, and the latter points to a permanent record of all KeplerMapper versions (we encourage you to cite the specific version you used). Example citations (for KeplerMapper 1.3.3):

van Veen et al., (2019). Kepler Mapper: A flexible Python implementation of the Mapper algorithm. Journal of Open Source Software, 4(42), 1315, https://doi.org/10.21105/joss.01315

Hendrik Jacob van Veen, Nathaniel Saul, Eargle, David, & Sam W. Mangham. (2019, October 14). Kepler Mapper: A flexible Python implementation of the Mapper algorithm (Version 1.3.3). Zenodo. http://doi.org/10.5281/zenodo.3485851