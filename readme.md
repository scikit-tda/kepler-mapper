# KeplerMapper <img align="right" width="40" height="40" src="http://i.imgur.com/axOG6GJ.jpg">

> Nature uses as little as possible of anything. - Johannes Kepler

This is a class containing a mapping algorithm in Python. KeplerMapper can be used for 
visualization of high-dimensional data and 3D point cloud data. 

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as first 
described in the paper "Topological Methods for the Analysis of High Dimensional 
Data Sets and 3D Object Recognition".

KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling algorithms.

## Parameters

**km.KeplerMapper**( cluster_algorithm=cluster.DBSCAN(eps=0.5,min_samples=3), nr_cubes=10, overlap_perc=0.1, scaler=preprocessing.MinMaxScaler(), color_function="distance_origin", link_local=False, verbose=1))

Parameter | Description
--- | ---
cluster_algorithm | Scikit-Learn API compatible clustering algorithm. The clustering algorithm to use for mapping. *Default = cluster.DBSCAN(eps=0.5,min_samples=3)*
nr_cubes | Int. The number of cubes/intervals to create. *Default = 10*
overlap_perc | Float. How much the cubes/intervals overlap (relevant for creating the edges). *Default = 0.1*
scaler | Scikit-Learn API compatible scaler. Scaler of the data applied before mapping. *Default = preprocessing.MinMaxScaler()*
color_function | String. The function to color nodes with. Currently only one function available/hardcoded. *Default = "distance_origin"*
link_local | Bool. Whether to link up local clusters. *Default = False*
verbose | Int. Verbosity of the mapper. *Default = 0*

**km.KeplerMapper().fit**(data)

Parameter | Description
--- | ---
data | Numpy Array. The data to fit the mapper on. *Required*

complex = **km.KeplerMapper().map**(data, dimension_index=0, dimension_name="")

Parameter | Description
--- | ---
data | Numpy Array. The data to map on. *Required*
dimension_index | Int. Index of dimension to map on. Not yet implemented is '-1' for all dimensions, or a list with multiple dimensions. *Default = 0*
dimension_name | String or Int. The human-readable name of the dimension to map(s) on. *Default = dimension_index*

complex = **km.KeplerMapper().visualize**(complex, path_html="mapper_visualization_output.html")

Parameter | Description
--- | ---
complex | Dictionary. The complex dictionary with nodes, edges and meta-information. *Required*
path_html | File path. Path where to output the .html file *Default = mapper_visualization_output.html*

## Install

The class is currently just one file. Simply dropping it in any directory which Python is able to import from should work.

## Required

These libraries are required to be installed for KeplerMapper to work:

* NumPy
* Scikit-Learn

KeplerMapper works on both Python 2.7 and Python 3+.

## External resources

These resources are loaded by the visualization output.

* Roboto Webfont (Google)
* D3.js (Mike Bostock)

## Usage

### Python code
```python
# Import the class
import km

# Some sample data
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=2000, noise=0.05, factor=0.3)

# Initialize
mapper = km.KeplerMapper(cluster_algorithm=cluster.DBSCAN(eps=0.1, min_samples=10), nr_cubes=25, overlap_perc=0.55, verbose=1)

# Fit to data
mapper.fit(data)

# Create dictionary called 'complex' with nodes, edges and meta-information
complex = mapper.map(data, dimension_index=0, dimension_name="X-axis")

# Visualize it
mapper.visualize(complex, path_html="make_circles_keplermapper_output.html", title="make_circles(n_samples=5000, noise=0.05, factor=0.3)")
```

### Console output
```
nr_cubes = 25

overlap_perc = 0.55

link_local = False

Clusterer = DBSCAN(algorithm='auto', eps=0.1, leaf_size=30, metric='euclidean',
    min_samples=10, p=None, random_state=None)
	
Scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

..Scaling

Mapping on data shaped (5000L, 2L)

There are 168 points in cube_0 with starting range 0.0
Found 1 clusters in cube_0

There are 327 points in cube_1 with starting range 0.04
Found 1 clusters in cube_1

There are 242 points in cube_2 with starting range 0.08
Found 1 clusters in cube_2

[... snipped ...]

There are 316 points in cube_22 with starting range 0.88
Found 1 clusters in cube_22

There are 139 points in cube_23 with starting range 0.92
Found 1 clusters in cube_23

There are 10 points in cube_24 with starting range 0.96
Found 0 clusters in cube_24


Wrote d3.js graph to 'make_circles_keplermapper_output.html'
```

### Visualization output

![Visualization](http://i.imgur.com/nS03AHV.png "Click for large")

Click here for an [interactive version](http://mlwave.github.io/tda/make_circles_keplermapper_output.html).

## References

> Mapper Algorithm
> "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition"
> Gurjeet Singh, Facundo Mémoli, and Gunnar Carlsson

http://www.ayasdi.com/wp-content/uploads/2015/02/Topological_Methods_for_the_Analysis_of_High_Dimensional_Data_Sets_and_3D_Object_Recognition.pdf

> Topological Data Analysis
> Stanford Seminar. "Topological Data Analysis: How Ayasdi used TDA to Solve Complex Problems"
> SF Data Mining. "Shape and Meaning."
> Anthony Bak

https://www.youtube.com/watch?v=x3Hl85OBuc0
https://www.youtube.com/watch?v=4RNpuZydlKY

> The shape of data
> "Conference Talk. The shape of data"
> Gunnar Carlsson

https://www.youtube.com/watch?v=kctyag2Xi8o

> Business Value, Problems, Algorithms, Computation and User Experience of TDA
> Data Driven NYC. "Making Data Work"
> Gurjeet Singh

https://www.youtube.com/watch?v=UZH5xJXJG2I

> Implementation details and sample data
> Python Mapper
> Daniel Müller et al.

http://danifold.net/mapper/index.html

> Applied Topology
> "Elementary Applied Topology"
> R. Ghrist

https://www.math.upenn.edu/~ghrist/notes.html

> Applied Topology
> "Qualitative data analysis"
> Community effort

http://appliedtopology.org/

> Single Linkage Clustering
> "Minimum Spanning Trees and Single Linkage Cluster Analysis"
> J. C. Gower, and G. J. S. Ross

http://www.cs.ucsb.edu/~veronika/MAE/mstSingleLinkage_GowerRoss_1969.pdf

> Clustering and Manifold Learning
> Scikit-learn: Machine Learning in Python
> Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.

http://scikit-learn.org/stable/modules/clustering.html
http://scikit-learn.org/stable/modules/manifold.html

> Color scales
> "Creating A Custom Hot to Cold Temperature Color Gradient for use with RRDTool"
> Dale Reagan

http://web-tech.ga-usa.com/2012/05/creating-a-custom-hot-to-cold-temperature-color-gradient-for-use-with-rrdtool/

> Design
> Material Design
> Google

https://design.google.com/

> Design
> Ayasdi Core Product Screenshots
> Ayasdi

http://www.ayasdi.com/product/core/

## Disclaimer

See disclaimer.txt for more. Basically this is a work in progress to familiarize myself with topological data analysis. The details of the algorithm implementations may be lacking. I'll gladly accept feedback and pull requests to make it more robust. You can contact me at info@mlwave.com or by opening an issue.
