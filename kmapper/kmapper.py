from __future__ import division

from collections import defaultdict
from datetime import datetime
import inspect
import itertools
import os
import sys
import warnings

from jinja2 import Environment, FileSystemLoader, Template
import numpy as np
from sklearn import cluster, preprocessing, manifold, decomposition
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.spatial import distance
from scipy.sparse import issparse

from .cover import Cover
from .nerve import GraphNerve
from .visuals import (
    init_color_function,
    format_meta,
    format_mapper_data,
    build_histogram,
    graph_data_distribution,
    colorscale_default,
)


# palette = [
#     "#0500ff",
#     "#0300ff",
#     "#0100ff",
#     "#0002ff",
#     "#0022ff",
#     "#0044ff",
#     "#0064ff",
#     "#0084ff",
#     "#00a4ff",
#     "#00a4ff",
#     "#00c4ff",
#     "#00e4ff",
#     "#00ffd0",
#     "#00ff83",
#     "#00ff36",
#     "#17ff00",
#     "#65ff00",
#     "#b0ff00",
#     "#fdff00",
#     "#FFf000",
#     "#FFdc00",
#     "#FFc800",
#     "#FFb400",
#     "#FFa000",
#     "#FF8c00",
#     "#FF7800",
#     "#FF6400",
#     "#FF5000",
#     "#FF3c00",
#     "#FF2800",
#     "#FF1400",
#     "#FF0000",
# ]


class KeplerMapper(object):
    """With this class you can build topological networks from (high-dimensional) data.

    1)   	Fit a projection/lens/function to a dataset and transform it.
                For instance "mean_of_row(x) for x in X"
    2)   	Map this projection with overlapping intervals/hypercubes.
                Cluster the points inside the interval
                (Note: we cluster on the inverse image/original data to lessen projection loss).
                If two clusters/nodes have the same members (due to the overlap), then:
                connect these with an edge.
    3)  	Visualize the network using HTML and D3.js.

    KM has a number of nice features, some which get forgotten.
        - project : Some projections it makes sense to use a distance matrix, such as knn_distance_#. Using `distance_matrix = <metric>` for a custom metric.
        - fit_transform : Applies a sequence of projections. Currently, this API is a little confusing and will be changed in the future. 
        - 


    """

    def __init__(self, verbose=0):
        """
        Inputs
        ======

        verbose: int, default is 0
            Logging level. Currently 3 levels (0,1,2) are supported.
                - for no logging, set `verbose=0`, 
                - for some logging, set `verbose=1`,
                - for complete logging, set `verbose=2`
        """

        # TODO: move as many of the arguments from fit_transform and map into here.
        self.verbose = verbose
        self.projection = None
        self.scaler = None
        self.cover = None

        if verbose > 0:
            print("KeplerMapper()")

    def project(
        self,
        X,
        projection="sum",
        scaler=preprocessing.MinMaxScaler(),
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
            Scaler of the data applied before mapping. Use None for no scaling. Default = preprocessing.MinMaxScaler() if None, do no scaling, else apply scaling to the projection. Default: Min-Max scaling

        distance_matrix : Either str or None
            If not None, then any of ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]. 
            If False do nothing, else create a squared distance matrix with the chosen metric, before applying the projection.

        Returns
        -------
        lens : Numpy Array
            projected data.

        Examples
        --------
        >>> projected_data = mapper.project(data, projection="sum", scaler=km.preprocessing.MinMaxScaler() )
        """

        # Sae original values off so they can be referenced by later functions in the pipeline
        self.inverse = X
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
        scaler=preprocessing.MinMaxScaler(),
        distance_matrix=False,
    ):
        """ Same as .project() but accepts lists for arguments so you can chain.

            Deprecated.

        """

        projections = projection
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
        clusterer=cluster.DBSCAN(eps=0.5, min_samples=3),
        cover=Cover(n_cubes=10, perc_overlap=0.1),
        nerve=GraphNerve(),
        precomputed=False,
        # These arguments are all deprecated
        overlap_perc=None,
        nr_cubes=None,
        coverer=None,
    ):
        """Apply Mapper algorithm on this projection and build a simplicial complex. Returns a dictionary with nodes and links.

        Parameters
        ----------
        lens: Numpy Array
            Lower dimensional representation of data. In general will be output of `fit_transform`.

        X: Numpy Array
            Original data or data to run clustering on. If `None`, then use `lens` as default.

        clusterer: Default: DBSCAN
            Scikit-learn API compatible clustering algorithm. Must provide `fit` and `predict`.

        cover: type kmapper.Cover
            Cover scheme for lens. Instance of kmapper.cover providing methods `define_bins` and `find_entries`.

        nerve: kmapper.Nerve
            Nerve builder implementing `__call__(nodes)` API

        precomputed : Boolean
            Tell Mapper whether the data that you are clustering on is a precomputed distance matrix. If set to
            `True`, the assumption is that you are also telling your `clusterer` that `metric='precomputed'` (which
            is an argument for DBSCAN among others), which 
            will then cause the clusterer to expect a square distance matrix for each hypercube. `precomputed=True` will give a square matrix
            to the clusterer to fit on for each hypercube.

        nr_cubes: Int (Deprecated)
            The number of intervals/hypercubes to create. Default = 10. (DeprecationWarning: define Cover explicitly in future versions)

        overlap_perc: Float (Deprecated)
            The percentage of overlap "between" the intervals/hypercubes. Default = 0.1. (DeprecationWarning: define Cover explicitly in future versions)

        Returns
        =======
        simplicial_complex : dict
            A dictionary with "nodes", "links" and "meta" information.

        Examples
        ========

        >>> simplicial_complex = mapper.map(lens, X=None, clusterer=cluster.DBSCAN(eps=0.5,min_samples=3), cover=km.Cover(n_cubes=[10,20], perc_overlap=0.4))

        >>>print(simplicial_complex["nodes"])
        >>>print(simplicial_complex["links"])
        >>>print(simplicial_complex["meta"])

        """

        start = datetime.now()

        nodes = defaultdict(list)
        meta = defaultdict(list)
        graph = {}

        # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
        if X is None:
            X = lens

        # Deprecation warnings
        if nr_cubes is not None or overlap_perc is not None:
            warnings.warn(
                "Deprecation Warning: Please supply km.Cover object. Explicitly passing in n_cubes/nr_cubes and overlap_perc will be deprecated in future releases. ",
                DeprecationWarning,
            )
        if coverer is not None:
            warnings.warn(
                "Deprecation Warning: coverer has been renamed to `cover`. Please use `cover` from now on.",
                DeprecationWarning,
            )

        # If user supplied nr_cubes, overlap_perc, or coverer, opt for those
        # TODO: remove this conditional after release in 1.2
        if coverer is not None:
            self.cover = coverer
        elif nr_cubes is not None or overlap_perc is not None:
            n_cubes = nr_cubes if nr_cubes else 10
            overlap_perc = overlap_perc if overlap_perc else 0.1
            self.cover = Cover(n_cubes=n_cubes, perc_overlap=overlap_perc)
        else:
            self.cover = cover

        if self.verbose > 0:
            print(
                "Mapping on data shaped %s using lens shaped %s\n"
                % (str(X.shape), str(lens.shape))
            )

        # Prefix'ing the data with an ID column
        ids = np.array([x for x in range(lens.shape[0])])
        lens = np.c_[ids, lens]
        X = np.c_[ids, X]

        # Cover scheme defines a list of elements
        bins = self.cover.define_bins(lens)

        # Algo's like K-Means, have a set number of clusters. We need this number
        # to adjust for the minimal number of samples inside an interval before
        # we consider clustering or skipping it.
        cluster_params = clusterer.get_params()

        min_cluster_samples = cluster_params.get(
            "n_clusters",
            cluster_params.get(
                "min_cluster_size", cluster_params.get("min_samples", 1)
            ),
        )

        if self.verbose > 1:
            print(
                "Minimal points in hypercube before clustering: %d"
                % (min_cluster_samples)
            )

        # Subdivide the projected data X in intervals/hypercubes with overlap
        if self.verbose > 0:
            bins = list(bins)  # extract list from generator
            total_bins = len(bins)
            print("Creating %s hypercubes." % total_bins)

        for i, cube in enumerate(bins):
            # Slice the hypercube:
            #  gather all entries in this element of the cover
            hypercube = self.cover.find_entries(lens, cube)

            if self.verbose > 1:
                print(
                    "There are %s points in cube %s/%s"
                    % (hypercube.shape[0], i, total_bins)
                )

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
                        "   > Found %s clusters.\n"
                        % (
                            np.unique(
                                cluster_predictions[cluster_predictions > -1]
                            ).shape[0]
                        )
                    )

                # TODO: I think this loop could be improved by turning inside out:
                #           - partition points according to each cluster
                # Now for every (sample id in cube, predicted cluster label)
                for idx, pred in np.c_[hypercube[:, 0], cluster_predictions]:
                    if pred != -1 and not np.isnan(pred):  # if not predicted as noise

                        # TODO: allow user supplied label
                        #   - where all those extra values necessary?
                        cluster_id = "cube{}_cluster{}".format(i, int(pred))

                        # Append the member id's as integers
                        nodes[cluster_id].append(int(idx))
                        meta[cluster_id] = {
                            "size": hypercube.shape[0],
                            "coordinates": cube,
                        }
            else:
                if self.verbose > 1:
                    print("Cube_%s is empty.\n" % (i))

        links, simplices = nerve(nodes)

        graph["nodes"] = nodes
        graph["links"] = links
        graph["simplices"] = simplices
        graph["meta_data"] = {
            "projection": self.projection if self.projection else "custom",
            "n_cubes": self.cover.n_cubes,
            "perc_overlap": self.cover.perc_overlap,
            "clusterer": str(clusterer),
            "scaler": str(self.scaler),
        }
        graph["meta_nodes"] = meta

        # Reporting
        if self.verbose > 0:
            self._summary(graph, str(datetime.now() - start))

        return graph

    def _summary(self, graph, time):
        # TODO: this summary is dependant on the type of Nerve being built.
        links = graph["links"]
        nodes = graph["nodes"]
        nr_links = sum(len(v) for k, v in links.items())

        print("\nCreated %s edges and %s nodes in %s." % (nr_links, len(nodes), time))

    def visualize(
        self,
        graph,
        color_function=None,
        custom_tooltips=None,
        custom_meta=None,
        path_html="mapper_visualization_output.html",
        title="Kepler Mapper",
        save_file=True,
        X=None,
        X_names=[],
        lens=None,
        lens_names=[],
        show_tooltips=True,
        nbins=10
    ):
        """Generate a visualization of the simplicial complex mapper output. Turns the complex dictionary into a HTML/D3.js visualization

        Parameters
        ----------
        graph : dict
            Simplicial complex output from the `map` method.

        path_html : String
            file name for outputing the resulting html.

        custom_meta: dict
            Render (key, value) in the Mapper Summary pane. 

        custom_tooltip: list or array like
            Value to display for each entry in the node. The cluster data pane will display entry for all values in the node. Default is index of data.

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

        show_tooltips: bool, default is True.
            If false, completely disable tooltips. This is useful when using output in space-tight pages or will display node data in custom ways.

        nbins: int, default is 10
            Number of bins shown in histogram of tooltip color distributions.

        Returns
        ------
        html: string
            Returns the same html that is normally output to `path_html`. Complete graph and data ready for viewing.

        Examples
        -------
        >>> mapper.visualize(simplicial_complex, path_html="mapper_visualization_output.html",
                            custom_meta={'Data': 'MNIST handwritten digits', 
                                         'Created by': 'Franklin Roosevelt'
                            }, )
        """

        # TODO:
        #   - Make color functions more intuitive. How do they even work?
        #   - Allow multiple color functions that can be toggled on and off.

        if not len(graph["nodes"]) > 0:
            raise Exception(
                "Visualize requires a mapper with more than 0 nodes. \nIt is possible that the constructed mapper could have been constructed with bad parameters. This can occasionally happens when using the default clustering algorithm. Try changing `eps` or `min_samples` in the DBSCAN clustering algorithm."
            )

        # Find the module absolute path and locate templates
        module_root = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(module_root))
        # Color function is a vector of colors?
        color_function = init_color_function(graph, color_function)

        mapper_data = format_mapper_data(
            graph, color_function, X, X_names, lens, lens_names, custom_tooltips, env, nbins
        )

        colorscale = colorscale_default

        histogram = graph_data_distribution(graph, color_function, colorscale)

        mapper_summary = format_meta(graph, custom_meta)

        # Find the absolute module path and the static files
        js_path = os.path.join(os.path.dirname(__file__), "static", "kmapper.js")
        with open(js_path, "r") as f:
            js_text = f.read()

        css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
        with open(css_path, "r") as f:
            css_text = f.read()

        # Render the Jinja template, filling fields as appropriate
        template = env.get_template("base.html").render(
            title=title,
            mapper_summary=mapper_summary,
            histogram=histogram,
            dist_label="Node",
            mapper_data=mapper_data,
            colorscale=colorscale,
            js_text=js_text,
            css_text=css_text,
            show_tooltips=True,
        )

        if save_file:
            with open(path_html, "wb") as outfile:
                if self.verbose > 0:
                    print("Wrote visualization to: %s" % (path_html))
                outfile.write(template.encode("utf-8"))

        return template

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
