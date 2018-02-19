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
from scipy.spatial import distance

from .cover import Cover
from .nerve import GraphNerve
from .visuals import init_color_function, format_meta, dict_to_json, color_function_distribution


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

    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.chunk_dist = []
        self.overlap_dist = []
        self.d = []
        self.projection = None
        self.scaler = None

    def fit_transform(self, X, projection="sum", scaler=preprocessing.MinMaxScaler(), distance_matrix=False):
        """Creates the projection/lens from a dataset. Input the data set. Specify a projection/lens type. Output the projected data/lens.


        Parameters
        ----------
        data : Numpy Array
            The data to fit a projection/lens to.

        projection :
            Projection parameter is either a string, a Scikit-learn class with fit_transform, like manifold.TSNE(), or a list of dimension indices. A string from ["sum", "mean", "median", "max", "min", "std", "dist_mean", "l2norm", "knn_distance_n"]. If using knn_distance_n write the number of desired neighbors in place of n: knn_distance_5 for summed distances to 5 nearest neighbors. Default = "sum".

        scaler :
            Scikit-Learn API compatible scaler. Scaler of the data applied before mapping. Use None for no scaling. Default = preprocessing.MinMaxScaler() if None, do no scaling, else apply scaling to the projection. Default: Min-Max scaling distance_matrix: False or any of: ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]. If False do nothing, else create a squared distance matrix with the chosen metric, before applying the projection.

        Returns
        -------
        lens : Numpy Array
            projected data.


        Example
        -------

        >>> projected_data = mapper.fit_transform(data, projection="sum", scaler=km.preprocessing.MinMaxScaler() )

        """
        self.inverse = X
        self.scaler = scaler
        self.projection = str(projection)
        self.distance_matrix = distance_matrix

        # If distance_matrix is a scipy.spatial.pdist string, we create a square distance matrix
        # from the vectors, before applying a projection.
        if self.distance_matrix in ["braycurtis",
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
                                    "yule"]:
            X = distance.squareform(distance.pdist(X, metric=distance_matrix))
            if self.verbose > 0:
                print("Created distance matrix, shape: %s, with distance metric `%s`" %
                      (X.shape, distance_matrix))

        # Detect if projection is a class (for scikit-learn)
        try:
            p = projection.get_params()
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

        # Detect if projection is a string (for standard functions)
        # TODO: test each one of these projections
        if isinstance(projection, str):
            if self.verbose > 0:
                print("\n..Projecting data using: %s" % (projection))
            # Stats lenses
            if projection == "sum":  # sum of row
                X = np.sum(X, axis=1).reshape((X.shape[0], 1))
            if projection == "mean":  # mean of row
                X = np.mean(X, axis=1).reshape((X.shape[0], 1))
            if projection == "median":  # mean of row
                X = np.median(X, axis=1).reshape((X.shape[0], 1))
            if projection == "max":  # max of row
                X = np.max(X, axis=1).reshape((X.shape[0], 1))
            if projection == "min":  # min of row
                X = np.min(X, axis=1).reshape((X.shape[0], 1))
            if projection == "std":  # std of row
                X = np.std(X, axis=1).reshape((X.shape[0], 1))
            if projection == "l2norm":
                X = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))

            if projection == "dist_mean":  # Distance of x to mean of X
                X_mean = np.mean(X, axis=0)
                X = np.sum(np.sqrt((X - X_mean)**2),
                           axis=1).reshape((X.shape[0], 1))

            if "knn_distance_" in projection:
                n_neighbors = int(projection.split("_")[2])
                if self.distance_matrix:  # We use the distance matrix for finding neighbors
                    X = np.sum(np.sort(X, axis=1)[:, :n_neighbors], axis=1).reshape(
                        (X.shape[0], 1))
                else:
                    from sklearn import neighbors
                    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
                    nn.fit(X)
                    X = np.sum(nn.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)[
                               0], axis=1).reshape((X.shape[0], 1))

        # Detect if projection is a list (with dimension indices)
        if isinstance(projection, list):
            if self.verbose > 0:
                print("\n..Projecting data using: %s" % (str(projection)))
            X = X[:, np.array(projection)]

        # Scaling
        if scaler is not None:
            if self.verbose > 0:
                print("\n..Scaling with: %s\n" % str(scaler))
            X = scaler.fit_transform(X)

        return X

    def map(self,
            projected_X,
            inverse_X=None,
            clusterer=cluster.DBSCAN(eps=0.5, min_samples=3),
            nr_cubes=None,
            overlap_perc=None,
            coverer=Cover(nr_cubes=10, overlap_perc=0.1),
            nerve=GraphNerve()):
        """Apply Mapper algorithm on this projection and build a simplicial complex. Returns a dictionary with nodes and links.

        Parameters
        ----------
        projected_X : Numpy Array
            Output from fit_transform

        inverse_X : Numpy Array
            Original data. If `None`, then use `projected_X` for clustering.

        clusterer:
            Scikit-learn API compatible clustering algorithm. Default: DBSCAN

        nr_cubes : Int
            The number of intervals/hypercubes to create. Default = 10. (DeprecationWarning: define Cover explicitly in future versions)

        overlap_perc : Float
            The percentage of overlap "between" the intervals/hypercubes. Default = 0.1. (DeprecationWarning: define Cover explicitly in future versions)

        coverer : kmapper.Cover
            Cover scheme for lens. Instance of kmapper.cover providing methods `define_bins` and `find_entries`.

        nerve : kmapper.Nerve
            Nerve builder implementing `__call__(nodes)` API

        Returns
        =======
        simplicial_complex : dict
            A dictionary with "nodes", "links" and "meta" information.

        Example
        =======

        >>> simplicial_complex = mapper.map(projected_X, inverse_X=None, clusterer=cluster.DBSCAN(eps=0.5,min_samples=3),nr_cubes=10, overlap_perc=0.1)

        >>>print(simplicial_complex["nodes"])
        >>>print(simplicial_complex["links"])
        >>>print(simplicial_complex["meta"])

        """

        start = datetime.now()

        nodes = defaultdict(list)
        meta = defaultdict(list)
        graph = {}

        # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
        if inverse_X is None:
            inverse_X = projected_X

        if nr_cubes is not None or overlap_perc is not None:
            # If user supplied nr_cubes or overlap_perc,
            # use old defaults instead of new Cover
            nr_cubes = nr_cubes if nr_cubes else 10
            overlap_perc = overlap_perc if overlap_perc else 0.1
            self.coverer = Cover(nr_cubes=nr_cubes,
                                 overlap_perc=overlap_perc)

            warnings.warn(
                "Explicitly passing in nr_cubes and overlap_perc will be deprecated in future releases. Please supply Cover object.", DeprecationWarning)
        else:
            self.coverer = coverer

        if self.verbose > 0:
            print("Mapping on data shaped %s using lens shaped %s\n" %
                  (str(inverse_X.shape), str(projected_X.shape)))

        # Prefix'ing the data with ID's
        ids = np.array([x for x in range(projected_X.shape[0])])
        projected_X = np.c_[ids, projected_X]
        inverse_X = np.c_[ids, inverse_X]

        # Cover scheme defines a list of elements
        bins = self.coverer.define_bins(projected_X)

        # Algo's like K-Means, have a set number of clusters. We need this number
        # to adjust for the minimal number of samples inside an interval before
        # we consider clustering or skipping it.
        cluster_params = clusterer.get_params()
        min_cluster_samples = cluster_params.get("n_clusters", 1)

        if self.verbose > 1:
            print("Minimal points in hypercube before clustering: %d" %
                  (min_cluster_samples))

        # Subdivide the projected data X in intervals/hypercubes with overlap
        if self.verbose > 0:
            bins = list(bins)  # extract list from generator
            total_bins = len(bins)
            print("Creating %s hypercubes." % total_bins)

        for i, cube in enumerate(bins):
            # Slice the hypercube:
            #  gather all entries in this element of the cover
            hypercube = self.coverer.find_entries(projected_X, cube)

            if self.verbose > 1:
                print("There are %s points in cube_%s / %s" %
                      (hypercube.shape[0], i, total_bins))

            # If at least min_cluster_samples samples inside the hypercube
            if hypercube.shape[0] >= min_cluster_samples:

                # Cluster the data point(s) in the cube, skipping the id-column
                # Note that we apply clustering on the inverse image (original data samples) that fall inside the cube.
                inverse_x = inverse_X[[int(nn) for nn in hypercube[:, 0]]]

                clusterer.fit(inverse_x[:, 1:])

                if self.verbose > 1:
                    print("Found %s clusters in cube_%s\n" % (
                        np.unique(clusterer.labels_[clusterer.labels_ > -1]).shape[0], i))

                # TODO: I think this loop could be improved by turning inside out:
                #           - partition points according to each cluster
                # Now for every (sample id in cube, predicted cluster label)
                for a in np.c_[hypercube[:, 0], clusterer.labels_]:
                    if a[1] != -1:  # if not predicted as noise

                        # TODO: allow user supplied label
                        #   - where all those extra values necessary?
                        cluster_id = "cube{}_cluster{}".format(i, int(a[1]))

                        # Append the member id's as integers
                        nodes[cluster_id].append(int(a[0]))
                        meta[cluster_id] = {
                            "size": hypercube.shape[0], "coordinates": cube}
            else:
                if self.verbose > 1:
                    print("Cube_%s is empty.\n" % (i))

        links, simplices = nerve(nodes)

        graph["nodes"] = nodes
        graph["links"] = links
        graph["simplices"] = simplices
        graph["meta_data"] = {
            "projection": self.projection if self.projection else "custom",
            "nr_cubes": self.coverer.nr_cubes,
            "overlap_perc": self.coverer.overlap_perc,
            "clusterer": str(clusterer),
            "scaler": str(self.scaler)
        }
        graph["meta_nodes"] = meta

        # Reporting
        if self.verbose > 0:
            self._summary(graph, str(datetime.now() - start))

        return graph

    def _summary(self, graph, time):
        # TODO: this summary is relevant to the type of Nerve being built.
        links = graph["links"]
        nodes = graph["nodes"]
        nr_links = sum(len(v) for k, v in links.items())

        print("\nCreated %s edges and %s nodes in %s." %
              (nr_links, len(nodes), time))

    def visualize(self,
                  graph,
                  color_function=None,
                  custom_tooltips=None,
                  custom_meta=None,
                  path_html="mapper_visualization_output.html",
                  title="My Data",
                  save_file=True,
                  inverse_X=None,
                  inverse_X_names=[],
                  projected_X=None,
                  projected_X_names=[]):
        """Generate a visualization of the simplicial complex mapper output. Turns the complex dictionary into a HTML/D3.js visualization

        Parameters
        ----------
        graph : dict
            Simplicial complex output from the `map` method.
        path_html : String
        file name for outputing the resulting html.

        Example
        -------

        >>> mapper.visualize(simplicial_complex, path_html="mapper_visualization_output.html")

        """

        color_function = init_color_function(graph, color_function)
        json_graph = dict_to_json(
            graph, color_function, inverse_X, inverse_X_names, projected_X, projected_X_names, custom_tooltips)
        color_distribution = color_function_distribution(
            graph, color_function)
        meta = format_meta(graph, custom_meta)


        # Find the absolute module path and the static files
        js_path = os.path.join(os.path.dirname(__file__), 'static', 'kmapper.js')
        with open(js_path, 'r') as myfile:
            js_text = myfile.read()
        css_path = os.path.join(os.path.dirname(__file__), 'static', 'style.css')
        with open(css_path, 'r') as myfile:
            css_text = myfile.read()

        # Find the module absolute path and locate templates
        module_root = os.path.join(os.path.dirname(__file__), 'templates')
        env = Environment(loader=FileSystemLoader(module_root))

        # Render the Jinja template, filling fields as appropriate
        template = env.get_template('base.html').render(
            title=title,
            meta=meta,
            color_distribution=color_distribution,
            json_graph=json_graph,
            js_text=js_text,
            css_text=css_text)

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
