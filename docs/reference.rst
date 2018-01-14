
Parameters
----------

Initialize
~~~~~~~~~~

.. code:: python

    mapper = km.KeplerMapper(verbose=1)

+-------------+------------------------------------------------------+
| Parameter   | Description                                          |
+=============+======================================================+
| verbose     | Int. Verbosity of the mapper. *Default = 0*          |
+-------------+------------------------------------------------------+

Fitting and transforming
~~~~~~~~~~~~~~~~~~~~~~~~

Input the data set. Specify a projection/lens type. Output the projected
data/lens.

.. code:: python

    projected_data = mapper.fit_transform(data, projection="sum",
                                          scaler=km.preprocessing.MinMaxScaler() )


+-----------------+------------------------------------------------+
|   Parameter     | Description                                    |
+=================+================================================+
| data            | Numpy Array. The  data to fit a projection/lens|
|                 | to. **Required**                               |
+-----------------+------------------------------------------------+
| projection      | Any of: list with dimension indices.           |
|                 | Scikit-learn API compatible manifold learner or|
|                 | dimensionality reducer. A string from ["sum",  |
|                 | "mean", "median", "max", "min", "std",         |
|                 | "dist_mean", "l2norm", "knn_distance_n"]. If   |
|                 | using knn_distance_n write the number of       |
|                 | desired neighbors in place of n: knn_distance_5|
|                 | for summed distances to 5 nearest neighbors.   |
|                 | **Default = "sum".**                           |
+-----------------+------------------------------------------------+
| scaler          | Scikit-Learn API compatible scaler. Scaler of  |
|                 | the data applied before mapping. Use None for  |
|                 | no scaling.                                    |
|                 | **Default = preprocessing.MinMaxScaler()**     |
+-----------------+------------------------------------------------+
| distance_matrix | False or any of: ["braycurtis", "canberra",    |
|                 | "chebyshev", "cityblock", "correlation",       |
|                 | "cosine", "dice", "euclidean", "hamming",      |
|                 | "jaccard", "kulsinski", "mahalanobis",         |
|                 | "matching", "minkowski", "rogerstanimoto",     |
|                 | "russellrao", "seuclidean", "sokalmichener",   |
|                 | "sokalsneath", "sqeuclidean", "yule"].         |
|                 | If False do nothing, else create a squared     |
|                 | distance matrix with the chosen metric, before |
|                 | applying the projection.                       |
+-----------------+------------------------------------------------+

Mapping
~~~~~~~

.. code:: python

    simplicial_complex = mapper.map(projected_X, inverse_X=None,
                                     clusterer=cluster.DBSCAN(eps=0.5,min_samples=3),
                                     nr_cubes=10, overlap_perc=0.1)

    print(simplicial_complex["nodes"])
    print(simplicial_complex["links"])
    print(simplicial_complex["meta"])

+----------------+------------------------------------------------------+
| Parameter      | Description                                          |
+================+======================================================+
| projected_X    | Numpy array. Output from fit_transform. **Required** |
+----------------+------------------------------------------------------+
| inverse_X      | Numpy array or None. When None, cluster on the       |
|                | projection, else cluster on the original data        |
|                | (inverse image).                                     |
+----------------+------------------------------------------------------+
| cluster        | Scikit-Learn API compatible clustering algorithm.    |
|                | The clustering algorithm to use for mapping.         |
|                | **Default = cluster.DBSCAN(eps=0.5,min_samples=3) ** |
+----------------+------------------------------------------------------+
| nr_cubes       | Int. The number of cubes/intervals to create.        |
|                | **Default = 10**                                     |
+----------------+------------------------------------------------------+
| overlap_perc   | Float. How much the cubes/intervals overlap          |
|                | (relevant for creating the edges). **Default = 0.1** |
+----------------+------------------------------------------------------+

Visualizing
~~~~~~~~~~~

.. code:: python

    mapper.visualize(topological_network,
                     path_html="mapper_visualization_output.html")

+--------------+-------------------------------------------------------+
| Parameter    | Description                                           |
+==============+=======================================================+
| graph        | Dict. The graph-dictionary with nodes, edges and      |
|              | meta-information. **Required**                        |
+--------------+-------------------------------------------------------+
| path_html    | File path. Path where to output the .html file        |
|              | **Default = mapper_visualization_output.html**        |
+--------------+-------------------------------------------------------+
