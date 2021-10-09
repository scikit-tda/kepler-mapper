from __future__ import division

from collections import defaultdict
from datetime import datetime
import inspect
import itertools
import os
import sys
import warnings

import numpy as np
from sklearn import cluster, preprocessing, manifold, decomposition
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.spatial import distance
from scipy.sparse import issparse, hstack

from .cover import Cover
from .nerve import GraphNerve
from .visuals import (
    _scale_color_values,
    _format_meta,
    _format_mapper_data,
    _build_histogram,
    _graph_data_distribution,
    colorscale_default,
    _render_d3_vis,
)
from .utils import deprecated_alias

# expose "cluster" to make examples and usage tidier
__all__ = ["KeplerMapper", "cluster"]


class KeplerMapper(object):
    """With this class you can build topological networks from (high-dimensional) data.

    1)          Fit a projection/lens/function to a dataset and transform it.
                For instance "mean_of_row(x) for x in X"
    2)          Map this projection with overlapping intervals/hypercubes.
                Cluster the points inside the interval
                (Note: we cluster on the inverse image/original data to lessen projection loss).
                If two clusters/nodes have the same members (due to the overlap), then:
                connect these with an edge.
    3)          Visualize the network using HTML and D3.js.

    KM has a number of nice features, some which get forgotten.
        - ``project``: Some projections it makes sense to use a distance matrix, such as knn_distance_#. Using ``distance_matrix = <metric>`` for a custom metric.
        - ``fit_transform``: Applies a sequence of projections. Currently, this API is a little confusing and might be changed in the future.



    """

    def __init__(self, verbose=0):
        """Constructor for KeplerMapper class.

        Parameters
        ===========

        verbose: int, default is 0
            Logging level. Currently 3 levels (0,1,2) are supported. For no logging, set `verbose=0`. For some logging, set `verbose=1`. For complete logging, set `verbose=2`.

        """

        # TODO: move as many of the arguments from fit_transform and map into here.
        self.verbose = verbose
        self.projection = None
        self.scaler = None
        self.cover = None

        if verbose > 0:
            print(self)

    def __repr__(self):
        return "KeplerMapper(verbose={})".format(self.verbose)

    def project(
        self,
        X,
        projection="sum",
        scaler="default:MinMaxScaler",
        distance_matrix=None,
    ):
        """Creates the projection/lens from a dataset. Input the data set. Specify a projection/lens type. Output the projected data/lens.

        Parameters
        ----------

        X : Numpy Array
            The data to fit a projection/lens to.

        projection :
            Projection parameter is either a string, a Scikit-learn class with fit_transform, like manifold.TSNE(), or a list of dimension indices. A string from ["sum", "mean", "median", "max", "min", "std", "dist_mean", "l2norm", "knn_distance_n"]. If using knn_distance_n write the number of desired neighbors in place of n: knn_distance_5 for summed distances to 5 nearest neighbors. Default = "sum".

        scaler : Scikit-Learn API compatible scaler.
            Scaler of the data applied after mapping. Use None for no scaling. Default = preprocessing.MinMaxScaler() if None, do no scaling, else apply scaling to the projection. Default: Min-Max scaling

        distance_matrix : Either str or None
            If not None, then any of ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"].
            If False do nothing, else create a squared distance matrix with the chosen metric, before applying the projection.

        Returns
        -------
        lens : Numpy Array
            projected data.

        Examples
        --------
        >>> # Project by taking the first dimension and third dimension
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     projection=[0,2]
        >>> )

        >>> # Project by taking the sum of row values
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     projection="sum"
        >>> )

        >>> # Do not scale the projection (default is minmax-scaling)
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     scaler=None
        >>> )

        >>> # Project by standard-scaled summed distance to 5 nearest neighbors
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     projection="knn_distance_5",
        >>>     scaler=sklearn.preprocessing.StandardScaler()
        >>> )

        >>> # Project by first two PCA components
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     projection=sklearn.decomposition.PCA()
        >>> )

        >>> # Project by first three UMAP components
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     projection=umap.UMAP(n_components=3)
        >>> )

        >>> # Project by L2-norm on squared Pearson distance matrix
        >>> X_projected = mapper.project(
        >>>     X_inverse,
        >>>     projection="l2norm",
        >>>     distance_matrix="pearson"
        >>> )

        >>> # Mix and match different projections
        >>> X_projected = np.c_[
        >>>     mapper.project(X_inverse, projection=sklearn.decomposition.PCA()),
        >>>     mapper.project(X_inverse, projection="knn_distance_5")
        >>> ]

        """

        # Sae original values off so they can be referenced by later functions in the pipeline
        self.inverse = X
        scaler = (
            preprocessing.MinMaxScaler() if scaler == "default:MinMaxScaler" else scaler
        )
        self.scaler = scaler
        self.projection = str(projection)
        self.distance_matrix = distance_matrix

        if self.verbose > 0:
            print("..Projecting on data shaped %s" % (str(X.shape)))

        # If distance_matrix is a scipy.spatial.pdist string, we create a square distance matrix
        # from the vectors, before applying a projection.
        if self.distance_matrix in [
            "braycurtis",
            "canberra",
            "chebyshev",
            "cityblock",
            "correlation",
            "cosine",
            "dice",
            "euclidean",
            "hamming",
            "jaccard",
            "kulsinski",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        ]:
            X = distance.squareform(distance.pdist(X, metric=distance_matrix))
            if self.verbose > 0:
                print(
                    "Created distance matrix, shape: %s, with distance metric `%s`"
                    % (X.shape, distance_matrix)
                )

        # Detect if projection is a class (for scikit-learn)
        try:
            p = projection.get_params()  # fail quickly
            reducer = projection
            if self.verbose > 0:
                try:
                    projection.set_params(**{"verbose": self.verbose})
                except:
                    pass
                print("\n..Projecting data using: \n\t%s\n" % str(projection))
            X = reducer.fit_transform(X)
        except:
            pass

        # What is this used for?
        if isinstance(projection, tuple):
            X = self._process_projection_tuple(projection)

        # Detect if projection is a string (for standard functions)
        # TODO: test each one of these projections
        if isinstance(projection, str):
            if self.verbose > 0:
                print("\n..Projecting data using: %s" % (projection))

            def dist_mean(X, axis=1):
                X_mean = np.mean(X, axis=0)
                X = np.sum(np.sqrt((X - X_mean) ** 2), axis=1)
                return X

            projection_funcs = {
                "sum": np.sum,
                "mean": np.mean,
                "median": np.median,
                "max": np.max,
                "min": np.min,
                "std": np.std,
                "l2norm": np.linalg.norm,
                "dist_mean": dist_mean,
            }

            if projection in projection_funcs.keys():
                X = projection_funcs[projection](X, axis=1).reshape((X.shape[0], 1))

            if "knn_distance_" in projection:
                n_neighbors = int(projection.split("_")[2])
                if (
                    self.distance_matrix
                ):  # We use the distance matrix for finding neighbors
                    X = np.sum(np.sort(X, axis=1)[:, :n_neighbors], axis=1).reshape(
                        (X.shape[0], 1)
                    )
                else:
                    from sklearn import neighbors

                    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
                    nn.fit(X)
                    X = np.sum(
                        nn.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)[
                            0
                        ],
                        axis=1,
                    ).reshape((X.shape[0], 1))

        # Detect if projection is a list (with dimension indices)
        if isinstance(projection, list):
            if self.verbose > 0:
                print("\n..Projecting data using: %s" % (str(projection)))
            X = X[:, np.array(projection)]

        # If projection produced sparse output, turn into a dense array
        if issparse(X):
            X = X.toarray()
            if self.verbose > 0:
                print("\n..Created projection shaped %s" % (str(X.shape)))

        # Scaling
        if scaler is not None:
            if self.verbose > 0:
                print("\n..Scaling with: %s\n" % str(scaler))
            X = scaler.fit_transform(X)

        return X

    def fit_transform(
        self,
        X,
        projection="sum",
        scaler="default:MinMaxScaler",
        distance_matrix=False,
    ):
        """Same as .project() but accepts lists for arguments so you can chain.

        Examples
        --------
        >>> # Stack / chain projections. You could do this manually,
        >>> # or pipeline with `.fit_transform()`. Works the same as `.project()`,
        >>> # but accepts lists. f(raw text) -> f(tfidf) -> f(isomap 100d) -> f(umap 2d)
        >>> projected_X = mapper.fit_transform(
        >>>     X,
        >>>     projections=[TfidfVectorizer(analyzer="char",
        >>>                                  ngram_range=(1,6),
        >>>                                  max_df=0.93,
        >>>                                  min_df=0.03),
        >>>                  manifold.Isomap(n_components=100,
        >>>                                  n_jobs=-1),
        >>>                  umap.UMAP(n_components=2,
        >>>                            random_state=1)],
        >>>     scalers=[None,
        >>>              None,
        >>>              preprocessing.MinMaxScaler()],
        >>>     distance_matrices=[False,
        >>>                        False,
        >>>                        False])

        """

        projections = projection
        scaler = (
            preprocessing.MinMaxScaler() if scaler == "default:MinMaxScaler" else scaler
        )
        scalers = scaler
        distance_matrices = distance_matrix

        # Turn single projection arguments into a pipeline
        if isinstance(projection, list) and isinstance(projection[0], int):
            projections = [projection]

        if not isinstance(projection, list):
            projections = [projection]

        # Turn single scaler arguments into a pipeline
        if not isinstance(scaler, list):
            scalers = [scaler]

        # Turn single distance matrix arguments into a pipeline
        if not isinstance(distance_matrix, list):
            distance_matrices = [distance_matrix]

        # set defaults to first list item, if not (correctly) set by the user
        if len(scalers) != len(projections):
            scalers = [scalers[0]] * len(projections)

        if len(distance_matrices) != len(projections):
            distance_matrices = [distance_matrices[0]] * len(projections)

        if self.verbose > 0:
            print("..Composing projection pipeline of length %s:" % (len(projections)))
            print("\tProjections: %s" % ("\n\t\t".join(map(str, projections))))
            print("\tDistance matrices: %s" % ("\n".join(map(str, distance_matrices))))
            print("\tScalers: %s" % ("\n".join(map(str, scalers))))

        # Pipeline Stack the projection functions
        lens = X
        for projection, scaler, distance_matrix in zip(
            projections, scalers, distance_matrices
        ):
            lens = self.project(
                lens,
                projection=projection,
                scaler=scaler,
                distance_matrix=distance_matrix,
            )

        return lens

    def map(
        self,
        lens,
        X=None,
        clusterer=None,
        cover=None,
        nerve=None,
        precomputed=False,
        remove_duplicate_nodes=False,
    ):
        """Apply Mapper algorithm on this projection and build a simplicial complex. Returns a dictionary with nodes and links.

        Parameters
        ----------
        lens: Numpy Array
            Lower dimensional representation of data. In general will be output of `fit_transform`.

        X: Numpy Array
            Original data or data to run clustering on. If `None`, then use `lens` as default. X can be a SciPy sparse matrix.

        clusterer: Default: DBSCAN
            Scikit-learn API compatible clustering algorithm. Must provide `fit` and `predict`.

        cover: kmapper.Cover
            Cover scheme for lens. Instance of kmapper.cover providing methods `fit` and `transform`.

        nerve: kmapper.Nerve
            Nerve builder implementing `__call__(nodes)` API

        precomputed : Boolean
            Tell Mapper whether the data that you are clustering on is a precomputed distance matrix. If set to
            `True`, the assumption is that you are also telling your `clusterer` that `metric='precomputed'` (which
            is an argument for DBSCAN among others), which
            will then cause the clusterer to expect a square distance matrix for each hypercube. `precomputed=True` will give a square matrix
            to the clusterer to fit on for each hypercube.

        remove_duplicate_nodes: Boolean
            Removes duplicate nodes before edges are determined. A node is considered to be duplicate
            if it has exactly the same set of points as another node.

        nr_cubes: Int

            .. deprecated:: 1.1.6

                define Cover explicitly in future versions

            The number of intervals/hypercubes to create. Default = 10.

        overlap_perc: Float
            .. deprecated:: 1.1.6

                define Cover explicitly in future versions

            The percentage of overlap "between" the intervals/hypercubes. Default = 0.1.



        Returns
        =======
        simplicial_complex : dict
            A dictionary with "nodes", "links" and "meta" information.

        Examples
        ========

        >>> # Default mapping.
        >>> graph = mapper.map(X_projected, X_inverse)

        >>> # Apply clustering on the projection instead of on inverse X
        >>> graph = mapper.map(X_projected)

        >>> # Use 20 cubes/intervals per projection dimension, with a 50% overlap
        >>> graph = mapper.map(X_projected, X_inverse,
        >>>                    cover=kmapper.Cover(n_cubes=20, perc_overlap=0.5))

        >>> # Use multiple different cubes/intervals per projection dimension,
        >>> # And vary the overlap
        >>> graph = mapper.map(X_projected, X_inverse,
        >>>                    cover=km.Cover(n_cubes=[10,20,5],
        >>>                                         perc_overlap=[0.1,0.2,0.5]))

        >>> # Use KMeans with 2 clusters
        >>> graph = mapper.map(X_projected, X_inverse,
        >>>     clusterer=sklearn.cluster.KMeans(2))

        >>> # Use DBSCAN with "cosine"-distance
        >>> graph = mapper.map(X_projected, X_inverse,
        >>>     clusterer=sklearn.cluster.DBSCAN(metric="cosine"))

        >>> # Use HDBSCAN as the clusterer
        >>> graph = mapper.map(X_projected, X_inverse,
        >>>     clusterer=hdbscan.HDBSCAN())

        >>> # Parametrize the nerve of the covering
        >>> graph = mapper.map(X_projected, X_inverse,
        >>>     nerve=km.GraphNerve(min_intersection=3))


        """

        start = datetime.now()

        clusterer = clusterer or cluster.DBSCAN(eps=0.5, min_samples=3)
        self.cover = cover or Cover(n_cubes=10, perc_overlap=0.1)
        nerve = nerve or GraphNerve()

        nodes = defaultdict(list)
        meta = defaultdict(list)
        graph = {}

        # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
        if X is None:
            X = lens

        if self.verbose > 0:
            print(
                "Mapping on data shaped %s using lens shaped %s\n"
                % (str(X.shape), str(lens.shape))
            )

        # Prefix'ing the data with an ID column
        ids = np.array([x for x in range(lens.shape[0])])
        lens = np.c_[ids, lens]
        if issparse(X):
            X = hstack([ids[np.newaxis].T, X], format="csr")
        else:
            X = np.c_[ids, X]

        # Cover scheme defines a list of elements
        bins = self.cover.fit(lens)

        # Algo's like K-Means, have a set number of clusters. We need this number
        # to adjust for the minimal number of samples inside an interval before
        # we consider clustering or skipping it.
        cluster_params = clusterer.get_params()

        min_cluster_samples = None
        for parameter in ["n_clusters", "min_cluster_size", "min_samples"]:
            value = cluster_params.get(parameter)
            if value and isinstance(value, int):
                min_cluster_samples = value
                break
        if not min_cluster_samples:
            min_cluster_samples = 2

        if self.verbose > 1:
            print(
                "Minimal points in hypercube before clustering: {}".format(
                    min_cluster_samples
                )
            )

        # Subdivide the projected data X in intervals/hypercubes with overlap
        if self.verbose > 0:
            bins = list(bins)  # extract list from generator
            total_bins = len(bins)
            print("Creating %s hypercubes." % total_bins)

        for i, hypercube in enumerate(self.cover.transform(lens)):

            # If at least min_cluster_samples samples inside the hypercube
            if hypercube.shape[0] >= min_cluster_samples:
                # Cluster the data point(s) in the cube, skipping the id-column
                # Note that we apply clustering on the inverse image (original data samples) that fall inside the cube.
                ids = [int(nn) for nn in hypercube[:, 0]]
                X_cube = X[ids]

                fit_data = X_cube[:, 1:]
                if precomputed:
                    fit_data = fit_data[:, ids]

                cluster_predictions = clusterer.fit_predict(fit_data)

                if self.verbose > 1:
                    print(
                        "   > Found %s clusters in hypercube %s."
                        % (
                            np.unique(
                                cluster_predictions[cluster_predictions > -1]
                            ).shape[0],
                            i,
                        )
                    )

                for pred in np.unique(cluster_predictions):
                    # if not predicted as noise
                    if pred != -1 and not np.isnan(pred):
                        cluster_id = "cube{}_cluster{}".format(i, int(pred))

                        nodes[cluster_id] = (
                            hypercube[:, 0][cluster_predictions == pred]
                            .astype(int)
                            .tolist()
                        )
            elif self.verbose > 1:
                print("Cube_%s is empty.\n" % (i))

        if remove_duplicate_nodes:
            nodes = self._remove_duplicate_nodes(nodes)

        links, simplices = nerve.compute(nodes)

        graph["nodes"] = nodes
        graph["links"] = links
        graph["simplices"] = simplices
        graph["meta_data"] = {
            "projection": self.projection if self.projection else "custom",
            "n_cubes": self.cover.n_cubes,
            "perc_overlap": self.cover.perc_overlap,
            "clusterer": str(clusterer),
            "scaler": str(self.scaler),
            "nerve_min_intersection": nerve.min_intersection
        }
        graph["meta_nodes"] = meta

        if self.verbose > 0:
            self._summary(graph, str(datetime.now() - start))

        return graph

    def _remove_duplicate_nodes(self, nodes):

        # invert node list and merge duplicate nodes
        deduped_items = defaultdict(list)
        for node_id, items in nodes.items():
            deduped_items[frozenset(items)].append(node_id)

        deduped_nodes = {
            "-".join(node_id_list): list(frozen_items)
            for frozen_items, node_id_list in deduped_items.items()
        }

        if self.verbose > 0:
            total_merged = len(nodes) - len(deduped_items)
            if total_merged:
                print("Merged {} duplicate nodes.\n".format(total_merged))
                print(
                    "Number of nodes before merger: {}; after merger: {}\n".format(
                        len(nodes), len(deduped_nodes)
                    )
                )
            else:
                print("No duplicate nodes found to remove.\n")

        return deduped_nodes

    def _summary(self, graph, time):
        # TODO: this summary is dependent on the type of Nerve being built.
        links = graph["links"]
        nodes = graph["nodes"]
        nr_links = sum(len(v) for k, v in links.items())

        print("\nCreated %s edges and %s nodes in %s." % (nr_links, len(nodes), time))

    @deprecated_alias(color_function="color_values")
    def visualize(
        self,
        graph,
        color_values=None,
        color_function_name=None,
        node_color_function="mean",
        colorscale=None,
        custom_tooltips=None,
        custom_meta=None,
        path_html="mapper_visualization_output.html",
        title="Kepler Mapper",
        save_file=True,
        X=None,
        X_names=None,
        lens=None,
        lens_names=None,
        nbins=10,
        include_searchbar=False,
        include_min_intersection_selector=False
    ):
        """Generate a visualization of the simplicial complex mapper output. Turns the complex dictionary into a HTML/D3.js visualization

        Parameters
        ----------
        graph : dict
            Simplicial complex output from the `map` method.

        color_function : list or 1d array
            .. deprecated:: 1.4.1
               Use `color_values` instead.

        color_values : list or 1d array, or list of 1d arrays
            color_values are sets (1d arrays) of values -- for each set, there should be
            one color value for each datapoint.

            These color values are used to compute the color value of a _node_ by applying `node_color_function` to
            the color values of each point within the node. The distribution of color_values for a given
            node can also be viewed in the visualization under the node details pane.

            A list of sets of color values (a list of 1d arrays) can be passed.
            If this is the case, then the visualization will have a toggle button
            for switching the visualization's currently active set of color values.

            If no color_values passed, then the data points' row positions are used as
            the set of color values.

        color_function_name : String or list
            A descriptor of the functions used to generate `color_values`.
            Will be used as labels in the visualization.
            If set, must be equal to the number of columns in color_values.

        node_color_function : String or 1d array, default is 'mean'
            Applied to the color_values of data points within a node to determine the color of the nodes.
            Will be applied column-wise to color_values.
            Must be a function available on numpy class object -- e.g., 'mean' => np.mean().

            If array, then 1d array of strings of np function names. Each node_color_function
            will be applied to each set of color_values (full permutation), and a toggle button will allow
            switching between the current active node_color_function for the visualization.

            See `visuals.py:_node_color_function()`

        colorscale : list
            Specify the colorscale to use. See visuals.colorscale_default.

        path_html : String
            file name for outputing the resulting html.

        custom_meta: dict
            Render (key, value) in the Mapper Summary pane.

        custom_tooltip: list or array like
            Value to display for each entry in the node. The cluster data pane will display entries for all values in the node. Default is index of data.

        save_file: bool, default is True
            Save file to `path_html`.

        X: numpy arraylike
            If supplied, compute statistics information about the original data source with respect to each node.

        X_names: list of strings
            Names of each variable in `X` to be displayed. If None, then display names by index.

        lens: numpy arraylike
            If supplied, compute statistics of each node based on the projection/lens

        lens_name: list of strings
            Names of each variable in `lens` to be displayed. In None, then display names by index.

        nbins: int, default is 10
            Number of bins shown in histogram of tooltip color distributions.

        include_searchbar: bool, default False
            Whether to include a search bar at the top of the visualization.

            The search functionality performs permits AND, OR, and EXACT
            methods, all against lowercased tooltips.

            * AND: the search query is split by whitespace. A data point's custom tooltip must
              match _each_ of the query terms in order to match overall. The base size of a node
              is multiplied by the number of datapoints matching the searchquery.
            * OR: the search query is split by whitespace. A data point's custom tooltip must
              match _any_ of the query terms in order to match overall. The base size of a node
              is multiplied by the number of datapoints matching the searchquery.
            * EXACT: A data point's custom tooltip must exactly match the query. Any nodes
              with a matching datapoint are set to glow.

            To reset any search-induced visual alterations, submit an empty search query.

        include_min_intersection_selector: bool, default False
            Whether to include an input to dynamically change the min_intersection
            for an edge to be drawn.

        Returns
        --------
        html: string
            Returns the same html that is normally output to `path_html`. Complete graph and data ready for viewing.

        Examples
        ---------

        >>> # Basic creation of a `.html` file at `kepler-mapper-output.html`
        >>> html = mapper.visualize(graph, path_html="kepler-mapper-output.html")

        >>> # Jupyter Notebook support
        >>> from kmapper import jupyter
        >>> html = mapper.visualize(graph, path_html="kepler-mapper-output.html")
        >>> jupyter.display(path_html="kepler-mapper-output.html")

        >>> # Customizing the output text
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     path_html="kepler-mapper-output.html",
        >>>     title="Fashion MNIST with UMAP",
        >>>     custom_meta={"Description":"A short description.",
        >>>                  "Cluster": "HBSCAN()"}
        >>> )

        >>> # Custom coloring data based on your 1d lens
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     color_values=lens
        >>> )

        >>> # Custom coloring data based on the first variable
        >>> cf = mapper.project(X, projection=[0])
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     color_values=cf
        >>> )

        >>> # Customizing the tooltips with binary target variables
        >>> X, y = split_data(df)
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     path_html="kepler-mapper-output.html",
        >>>     title="Fashion MNIST with UMAP",
        >>>     custom_tooltips=y
        >>> )

        >>> # Customizing the tooltips with html-strings: locally stored images of an image dataset
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     path_html="kepler-mapper-output.html",
        >>>     title="Fashion MNIST with UMAP",
        >>>     custom_tooltips=np.array(
        >>>             ["<img src='img/%s.jpg'>"%i for i in range(inverse_X.shape[0])]
        >>>     )
        >>> )

        >>> # Using multiple datapoint color functions
        >>> # Uses a two-dimensional lens, so two `color_function_name`s are required
        >>> lens = np.c_[isolation_forest_lens, l2_norm_lens]
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     path_html="breast-cancer-multiple-color-functions.html",
        >>>     title="Wisconsin Breast Cancer Dataset",
        >>>     color_values=lens,
        >>>     color_function_name=['Isolation Forest', 'L2-norm']
        >>> )

        >>> # Using multiple node color functions
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     path_html="breast-cancer-multiple-color-functions.html",
        >>>     title="Wisconsin Breast Cancer Dataset",
        >>>     node_color_function=['mean', 'std', 'median', 'max']
        >>> )

        >>> # Combining both multiple datapoint color functions and multiple node color functions
        >>> lens = np.c_[isolation_forest_lens, l2_norm_lens]
        >>> html = mapper.visualize(
        >>>     graph,
        >>>     path_html="breast-cancer-multiple-color-functions.html",
        >>>     title="Wisconsin Breast Cancer Dataset",
        >>>     color_values=lens,
        >>>     color_function_name=['Isolation Forest', 'L2-norm']
        >>>     node_color_function=['mean', 'std', 'median', 'max']
        >>> )

        """
        if colorscale is None:
            colorscale = colorscale_default

        if X_names is None:
            X_names = []

        if lens_names is None:
            lens_names = []

        if not len(graph["nodes"]) > 0:
            raise Exception(
                "Visualize requires a mapper with more than 0 nodes. \nIt is possible that the constructed mapper could have been constructed with bad parameters. This can occasionally happens when using the default clustering algorithm. Try changing `eps` or `min_samples` in the DBSCAN clustering algorithm."
            )

        if color_function_name is None:
            color_function_name = []
        elif isinstance(color_function_name, str):
            color_function_name = [color_function_name]

        if isinstance(node_color_function, str):
            node_color_function = [node_color_function]

        for _node_color_function_name in node_color_function:
            try:
                getattr(np, _node_color_function_name)
            except AttributeError as e:
                raise AttributeError(
                    "Invalid `node_color_function` {}, must be a function available on `numpy` class.".format(
                        _node_color_function_name
                    )
                ) from e

        if color_values is None:
            # We generate default `color_values` based on data row order
            n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
            color_values = np.arange(n_samples)
            if not len(color_function_name):
                color_function_name = ["Row number"]
            else:
                # `color_function_name` was not None, while `color_values` was None
                #
                # This is okay, as long as there's only one entry for `color_function_name`.
                # If this is the case, then that will be used to name the default
                # `color_values` based on row order. But we will raise a warning.

                if len(color_function_name) == 1:
                    warnings.warn(
                        "`color_function_name` was set -- however, no `color_values` were passed, so default color_values were computed based on row order, and the passed `color_function_name` will be set as their label. This may be unexpected."
                    )
                else:
                    raise Exception(
                        "More than one `color_function_name` was set, while `color_values` was not set. If `color_values` was not set, then only one `color_function_name` can be passed. Refusing to proceed."
                    )
        else:
            color_values = np.array(color_values)
            # test whether we have a color_function_name for each color_value vector
            if color_values.ndim == 1:
                num_color_value_vectors = 1
            else:
                num_color_value_vectors = color_values.shape[1]
            num_color_function_names = len(color_function_name)
            if num_color_value_vectors != num_color_function_names:
                raise Exception(
                    "{} `color_function_names` values found, but {} columns found in color_values. Must be equal.".format(
                        num_color_function_names, num_color_value_vectors
                    )
                )

        color_values = _scale_color_values(color_values)

        mapper_data = _format_mapper_data(
            graph,
            color_values,
            node_color_function,
            X,
            X_names,
            lens,
            lens_names,
            custom_tooltips,
            nbins,
            colorscale=colorscale,
        )

        histogram = []
        for _node_color_function_name in node_color_function:
            _histogram = _graph_data_distribution(
                graph, color_values, _node_color_function_name, colorscale
            )
            if np.array(_histogram).ndim == 1:
                _histogram = [_histogram]  # javascript will expect the histogram
                # array to be indexed for the number of
                # node_color_functions first, and second
                # for the number of color_functions
            histogram.append(_histogram)

        mapper_summary = _format_meta(
            graph, color_function_name, node_color_function, custom_meta
        )

        html = _render_d3_vis(
            title, mapper_summary, histogram, mapper_data, colorscale, include_searchbar, include_min_intersection_selector
        )

        if save_file:
            with open(path_html, "wb") as outfile:
                if self.verbose > 0:
                    print("Wrote visualization to: %s" % (path_html))
                outfile.write(html.encode("utf-8"))

        return html

    def data_from_cluster_id(self, cluster_id, graph, data):
        """Returns the original data of each cluster member for a given cluster ID

        Parameters
        ----------
        cluster_id : String
            ID of the cluster.
        graph : dict
            The resulting dictionary after applying map()
        data : Numpy Array
            Original dataset. Accepts both 1-D and 2-D array.

        Returns
        -------
        entries:
            rows of cluster member data as Numpy array.

        """
        if cluster_id in graph["nodes"]:
            cluster_members = graph["nodes"][cluster_id]
            cluster_members_data = data[cluster_members]
            return cluster_members_data
        else:
            return np.array([])

    def _process_projection_tuple(self, projection):
        # Detect if projection is a tuple (for prediction functions)
        # TODO: multi-label models
        # TODO: infer binary classification and select positive class preds
        # TODO: turn into smaller functions for better tests and complexity

        # TODO: this seems like outside the purview of mapper. Can we add something like Mapper utils that can do this?

        def blend(X_blend, pred_fun, folder, X_data, y):
            for train_index, test_index in folder.split(X_data, y):
                fold_X_train = X_data[train_index]
                fold_y_train = y[train_index]
                fold_X_test = X_data[test_index]
                fold_y_test = y[test_index]
                model.fit(fold_X_train, fold_y_train)
                fold_preds = pred_fun(fold_X_test)
                X_blend[test_index] = fold_preds

            return X_blend

        # If projection was passed without ground truth
        # assume we are predicting a fitted model on a test set
        if len(projection) == 2:
            model, X_data = projection
            # Are we dealing with a classifier or a regressor?
            estimator_type = getattr(model, "_estimator_type", None)
            if estimator_type == "classifier":
                # classifier probabilities
                X_blend = model.predict_proba(X_data)
            elif estimator_type == "regressor":
                X_blend = model.predict(X_data)
            else:
                warnings.warn("Unknown estimator type for: %s" % (model))

        # If projection is passed with ground truth do 5-fold stratified
        # cross-validation, saving the out-of-fold predictions.
        # this is called "Stacked Generalization" (see: Wolpert 1992)
        elif len(projection) == 3:
            model, X_data, y = projection
            estimator_type = getattr(model, "_estimator_type", None)

            if estimator_type == "classifier":
                X_blend = np.zeros((X_data.shape[0], np.unique(y).shape[0]))
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1729)

                blend(X_blend, model.predict_proba, skf, X_data, y)
            elif estimator_type == "regressor":
                X_blend = np.zeros(X_data.shape[0])
                kf = KFold(n_splits=5, shuffle=True, random_state=1729)
                blend(X_blend, model.predict, kf, X_data, y)
            else:
                warnings.warn("Unknown estimator type for: %s" % (model))
        else:
            # Warn for malformed input and provide help to avoid it.
            warnings.warn(
                "Passing a model function should be"
                + "(model, X) or (model, X, y)."
                + "Instead got %s" % (str(projection))
            )
        # Reshape 1-D arrays (regressor outputs) to 2-D arrays
        if X_blend.ndim == 1:
            X_blend = X_blend.reshape((X_blend.shape[0], 1))

        X = X_blend

        return X
