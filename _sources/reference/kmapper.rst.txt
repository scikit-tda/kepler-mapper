
KeplerMapper
--------------

The `KeplerMapper` class provides three main functions: `fit_transform, map, and visualize`. These provide the bulk of the functionality for constructing a Mapper graph. 

- `fit_transform` generates the lens (or filter function) for the data set,
- `map` builds the graph from the lens and data. It also uses a cover and clustering object.
- `visualize` constructs a visualization of the graph. We also provide other methods for visualizing, see the `Visuals documentation page`_


.. automodule:: kmapper


.. autoclass:: KeplerMapper
    :members: fit_transform, map, visualize, project, data_from_cluster_id
    
.. _Visuals documentation page: visuals.html