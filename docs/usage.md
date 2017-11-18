
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

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
```

### Console output
```
..Projecting data using: [0, 1]

..Scaling with: MinMaxScaler(copy=True, feature_range=(0, 1))

Mapping on data shaped (5000L, 2L) using dimensions

Creating 100 hypercubes.

created 86 edges and 57 nodes in 0:00:03.614000.

Wrote d3.js graph to 'make_circles_keplermapper_output.html'
```

### Visualization output

![Visualization](http://i.imgur.com/i3cqQVr.png "Click for large")

Click here for an [interactive version](http://mlwave.github.io/tda/make_circles_keplermapper_output2.html).
Click here for an older [interactive version](http://mlwave.github.io/tda/make_circles_keplermapper_output.html).



The class is currently just one file. Simply dropping `kmapper/kmapper.py` in any directory which Python is able to import from should work.


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
projection | Any of: list with dimension indices. Scikit-learn API compatible manifold learner or dimensionality reducer. A string from ["sum","mean","median","max","min","std","dist_mean","l2norm","knn_distance_n"]. If using `knn_distance_n` write the number of desired neighbors in place of `n`: `knn_distance_5` for summed distances to 5 nearest neighbors. *Default = "sum"*.                                
scaler | Scikit-Learn API compatible scaler. Scaler of the data applied before mapping. Use `None` for no scaling. *Default = preprocessing.MinMaxScaler()*
distance_matrix | `False` or any of: ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]. If `False` do nothing, else create a squared distance matrix with the chosen metric, before applying the projection.

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
