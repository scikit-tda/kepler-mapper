# KeplerMapper <img align="right" width="40" height="40" src="http://i.imgur.com/axOG6GJ.jpg">

> Nature uses as little as possible of anything. - Johannes Kepler

This is a class containing a mapping algorithm in Python. KeplerMapper can be used for 
visualization of high-dimensional data and 3D point cloud data. 

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as first 
described in the paper "Topological Methods for the Analysis of High Dimensional 
Data Sets and 3D Object Recognition".

KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling algorithms.

## Usage

### Python code
```python
# Import the class
import km

# Some sample data
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1]) # X-Y axis

# Create dictionary called 'complex' with nodes, edges and meta-information
complex = mapper.map(projected_data, data, nr_cubes=10)

# Visualize it
mapper.visualize(complex, path_html="make_circles_keplermapper_output.html", 
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
```

### Console output
```
..Projecting data using: [0, 1]

..Scaling with: MinMaxScaler(copy=True, feature_range=(0, 1))

Mapping on data shaped (5000L, 2L) using dimensions

Creating 1000 hypercubes.

created 86 edges and 57 nodes in 0:00:03.614000.

Wrote d3.js graph to 'make_circles_keplermapper_output.html'
```

### Visualization output

![Visualization](http://i.imgur.com/i3cqQVr.png "Click for large")

Click here for an [interactive version](http://mlwave.github.io/tda/make_circles_keplermapper_output2.html).
Click here for an older [interactive version](http://mlwave.github.io/tda/make_circles_keplermapper_output.html).

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

## Parameters

### Initialize

```python
mapper = km.KeplerMapper(verbose=1)
```

Parameter | Description
--- | ---
verbose | Int. Verbosity of the mapper. *Default = 0*

### Fitting and transforming
Input the data set. Specify a projection/lens type. Output the projected data/lens.

```python
projected_data = mapper.fit_transform(data, projection="sum", 
									  scaler=km.preprocessing.MinMaxScaler() )
```

Parameter | Description
--- | ---
data | Numpy Array. The data to fit a projection/lens to. *Required*
projection | Any of: list with dimension indices. Scikit-learn API compatible manifold learner or dimensionality reducer. A string from ["sum","mean","median","max","min","std","dist_mean"]. *Default = "sum"*                               
scaler | Scikit-Learn API compatible scaler. Scaler of the data applied before mapping. Use `None` for no scaling. *Default = preprocessing.MinMaxScaler()*

### Mapping

```python
topological_network = mapper.map(projected_X, inverse_X=None, 
								 clusterer=cluster.DBSCAN(eps=0.5,min_samples=3), 
								 nr_cubes=10, overlap_perc=0.1)

print(topological_network["nodes"])
print(topological_network["links"])
print(topological_network["meta"])
```

Parameter | Description
--- | ---
projected_X | Numpy array. Output from fit_transform. *Required*
inverse_X | Numpy array or `None`. When `None`, cluster on the projection, else cluster on the original data (inverse image).
clusterer | Scikit-Learn API compatible clustering algorithm. The clustering algorithm to use for mapping. *Default = cluster.DBSCAN(eps=0.5,min_samples=3)*
nr_cubes | Int. The number of cubes/intervals to create. *Default = 10*
overlap_perc | Float. How much the cubes/intervals overlap (relevant for creating the edges). *Default = 0.1*

### Visualizing

```python
mapper.visualize(topological_network, 
				 path_html="mapper_visualization_output.html")
```

Parameter | Description
--- | ---
topological_network | Dict. The `topological_network`-dictionary with nodes, edges and meta-information. *Required*
path_html | File path. Path where to output the .html file *Default = mapper_visualization_output.html*
title | String. Document title for use in the outputted .html. *Default = "My Data"*
graph_link_distance | Int. Global length of links between nodes. Use less for larger graphs. *Default = 30*
graph_charge | Int. The charge between nodes. Use less negative charge for larger graphs. *Default = -120*
graph_gravity | Float. A weak geometric constraint similar to a virtual spring connecting each node to the center of the layout's size. Don't you set to negative or it's turtles all the way up. *Default = 0.1*
custom_tooltips | NumPy Array. Create custom tooltips for all the node members. You could use the target labels `y` for this. Use `None` for standard tooltips. *Default = None*.
show_title | Bool. Whether to show the title. *Default = True*
show_meta | Bool. Whether to show meta information, like the overlap percentage and the clusterer used. *Default = True*
show_tooltips | Bool. Whether to show the tooltips on hover. *Default = True*
width_html | Int. Size in pixels of the graph canvas width. *Default = 0 (full screen width)*
height_html | Int. Size in pixels of the graph canvas height. *Default = 0 (full screen height)*

## Examples

### 3D-point cloud

Check the `examples` directory for more.

![Visualization](http://i.imgur.com/OQqHt9R.png "Click for large")

### Very noisy datasets

Check the `examples\makecircles` directory for code

![Visualization](http://i.imgur.com/OmETfe5.png "Click for large")

### Dimensionality reduction

t-SNE on 4K images of MNIST dataset.

![Visualization](http://i.imgur.com/eRa9sMH.png "Click for large")

## References

> Mapper Algorithm<br/>
> "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition"<br/>
> Gurjeet Singh, Facundo Mémoli, and Gunnar Carlsson

http://www.ayasdi.com/wp-content/uploads/2015/02/Topological_Methods_for_the_Analysis_of_High_Dimensional_Data_Sets_and_3D_Object_Recognition.pdf

> Topological Data Analysis<br/>
> Stanford Seminar. "Topological Data Analysis: How Ayasdi used TDA to Solve Complex Problems"<br/>
> SF Data Mining. "Shape and Meaning."<br/>
> Anthony Bak

https://www.youtube.com/watch?v=x3Hl85OBuc0<br/>
https://www.youtube.com/watch?v=4RNpuZydlKY

> Projection vs. Inverse image & Examples<br/>
> MLconf ATL. Topological Learning with Ayasdi<br/>
> Allison Gilmore

https://www.youtube.com/watch?v=cJ8W0ASsnp0

> The shape of data<br/>
> "Conference Talk. The shape of data"<br/>
> Topology and Data<br/>
> Gunnar Carlsson

https://www.youtube.com/watch?v=kctyag2Xi8o
http://www.ams.org/images/carlsson-notes.pdf

> Business Value, Problems, Algorithms, Computation and User Experience of TDA<br/>
> Data Driven NYC. "Making Data Work"<br/>
> Gurjeet Singh

https://www.youtube.com/watch?v=UZH5xJXJG2I

> Implementation details and sample data<br/>
> Python Mapper<br/>
> Daniel Müllner and Aravindakshan Babu

http://danifold.net/mapper/index.html

> Applied Topology<br/>
> "Elementary Applied Topology"<br/>
> R. Ghrist

https://www.math.upenn.edu/~ghrist/notes.html

> Applied Topology<br/>
> "Qualitative data analysis"<br/>
> Community effort

http://appliedtopology.org/

> Single Linkage Clustering<br/>
> "Minimum Spanning Trees and Single Linkage Cluster Analysis"<br/>
> J. C. Gower, and G. J. S. Ross

http://www.cs.ucsb.edu/~veronika/MAE/mstSingleLinkage_GowerRoss_1969.pdf

> Clustering and Manifold Learning<br/>
> Scikit-learn: Machine Learning in Python<br/>
> Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.

http://scikit-learn.org/stable/modules/clustering.html<br/>
http://scikit-learn.org/stable/modules/manifold.html

> Force-directed Graphing/Clustering<br/>
> Force-directed Graphs<br/>
> Mike Bostock, Tim Dwyer, Thomas Jakobsen

http://bl.ocks.org/mbostock/4062045

> Graphing<br/>
> Grapher<br/>
> Cindy Zhang, Danny Cochran, Diana Suvorova, Curtis Mitchell

https://github.com/ayasdi/grapher

> Color scales<br/>
> "Creating A Custom Hot to Cold Temperature Color Gradient for use with RRDTool"<br/>
> Dale Reagan

http://web-tech.ga-usa.com/2012/05/creating-a-custom-hot-to-cold-temperature-color-gradient-for-use-with-rrdtool/

> Design<br/>
> Material Design<br/>
> Google

https://design.google.com/

> Design<br/>
> Ayasdi Core Product Screenshots<br/>
> Ayasdi

http://www.ayasdi.com/product/core/

## Disclaimer

See disclaimer.txt for more. Basically this is a work in progress to familiarize myself with topological data analysis. The details of the algorithm implementations may be lacking. I'll gladly accept feedback and pull requests to make it more robust. You can contact me at info@mlwave.com or by opening an issue.