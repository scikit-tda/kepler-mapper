from __future__ import division
import sys
import inspect
import json
import itertools
from collections import defaultdict
from datetime import datetime
import warnings

import numpy as np
from sklearn import cluster, preprocessing, manifold, decomposition
from scipy.spatial import distance

from nerve import GraphNerve

class Cover():
    """Helper class that defines the default covering scheme

    functions
    ---------
    cubes:          @property, returns an iterable of all bins in the cover.
    find_entries:   Find all entries in the input data that are in the given cube.
    """

    def __init__(self, nr_cubes=10, overlap_perc=0.2):
        self.nr_cubes = nr_cubes
        self.overlap_perc = overlap_perc

    def define_bins(self, data):
        """
        Helper function to get origin coordinates for our intervals/hypercubes
        Useful for looping no matter the number of cubes or dimensions
        Example:   	if there are 4 cubes per dimension and 3 dimensions
                        return the bottom left (origin) coordinates of 64 hypercubes,
                        as a sorted list of Numpy arrays

        This function must assume that the first column of data are indices.
        """

        indexless_data = data[:, 1:]
        bounds = (np.min(indexless_data, axis=0),
                  np.max(indexless_data, axis=0))

        # We chop up the min-max column ranges into 'nr_cubes' parts
        self.chunk_dist = (bounds[1] - bounds[0]) / self.nr_cubes

        # We calculate the overlapping windows distance
        self.overlap_dist = self.overlap_perc * self.chunk_dist

        # We find our starting point
        self.d = bounds[0]

        # Use a dimension index array on the projected X
        # (For now this uses the entire dimensionality, but we keep for experimentation)
        self.di = np.array(range(1, data.shape[1]))
        self.nr_dimensions = len(self.di)

        if type(self.nr_cubes) is not list:
            cubes = [self.nr_cubes] * self.nr_dimensions
        else:
            assert len(self.nr_cubes) == self.nr_dimensions, "There are {} ({}) dimensions specified but {} dimensions needing specification. If you supply specific number of cubes for each dimension, please supply the correct number.".format(
                len(self.nr_cubes), self.nr_cubes, self.nr_dimensions)
            cubes = self.nr_cubes

        coordinates = map(np.asarray, itertools.product(
            *(range(i) for i in cubes)))

        return coordinates

    def find_entries(self, data, cube, verbose=0):
        """Find all entries in data that are in the given cube

        Input:      data, cube (an item from the list of cubes provided by `cover.cubes`)
        Output:     all entries in data that are in cube.
        """

        chunk = self.chunk_dist
        overlap = self.overlap_dist
        lower_bound = self.d + (cube * chunk)
        upper_bound = lower_bound + chunk + overlap

        # Slice the hypercube
        entries = (data[:, self.di] >= lower_bound) & \
                  (data[:, self.di] < upper_bound)

        hypercube = data[np.invert(np.any(entries == False, axis=1))]

        return hypercube


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

    functions
    ---------
    fit_transform:   Create a projection (lens) from a dataset
    map:         	Apply Mapper algorithm on this projection and build a simplicial complex
    visualize:    	Turns the complex dictionary into a HTML/D3.js visualization
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.chunk_dist = []
        self.overlap_dist = []
        self.d = []
        self.projection = None
        self.scaler = None

    def fit_transform(self, X, projection="sum", scaler=preprocessing.MinMaxScaler(), distance_matrix=False):
        """Creates the projection/lens from X.

        Input:      X. Input features as a numpy array.
        Output:     projected_X. original data transformed to a projection (lens).

        parameters
        ----------
        projection:   Projection parameter is either a string,
                      a scikit class with fit_transform, like manifold.TSNE(),
                      or a list of dimension indices.
        scaler:       if None, do no scaling, else apply scaling to the projection
                      Default: Min-Max scaling
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
        """This maps the data to a simplicial complex. Returns a dictionary with nodes and links.

        Input:    projected_X. A Numpy array with the projection/lens.
        Output:    complex. A dictionary with "nodes", "links" and "meta information"

        parameters
        ----------
        projected_X:    projected_X. A Numpy array with the projection/lens.
                        Required.
        inverse_X:      Numpy array or None. If None then the projection itself
                        is used for clustering.
        clusterer:      Scikit-learn API compatible clustering algorithm.
                        Default: DBSCAN
        nr_cubes:       Int. The number of intervals/hypercubes to create.
                        (DeprecationWarning, define Cover explicitly in future versions)
        overlap_perc:   Float. The percentage of overlap "between" the intervals/hypercubes.
                        (DeprecationWarning, define Cover explicitly in future versions)
        coverer:        Cover scheme for lens. Instance of kmapper. Cover providing
                        methods `define_bins` and `find_entries`.
        nerve           Nerve builder implementing __call__(nodes) API
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

        # TODO: NetworkX to Graph Conversion

        def _init_color_function(graph, color_function):
            # If no color_function provided we color by row order in data set
            # Reshaping to 2-D array is required for sklearn 0.19
            n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
            if color_function is None:
                color_function = np.arange(n_samples).reshape(-1, 1)
            else:
                color_function = color_function.reshape(-1, 1)
            # MinMax Scaling to be friendly to non-scaled input.   
            scaler = preprocessing.MinMaxScaler()
            color_function = scaler.fit_transform(color_function).ravel()
            return color_function

        def _format_cluster_statistics(member_ids, inverse_X, inverse_X_names):
            cluster_stats = ""
            if inverse_X is not None:
                # List vs. numpy handling: cast to numpy array
                if isinstance(inverse_X_names, list):
                    inverse_X_names = np.array(inverse_X_names)
                # Defaults when providing no inverse_X_names
                if inverse_X_names.shape[0] == 0:
                    inverse_X_names = np.array(["f_%s"%(i) for i in range(
                                                              inverse_X.shape[1])])

                cluster_X_mean = np.mean(inverse_X[member_ids], axis=0)
                inverse_X_mean = np.mean(inverse_X, axis=0)
                inverse_X_std = np.std(inverse_X, axis=0)
                above_mean = cluster_X_mean > inverse_X_mean
                std_m = np.sqrt((cluster_X_mean - inverse_X_mean)**2) / inverse_X_std

                stats = sorted([(s,f,i,c,a,v) for s,f,i,c,a,v in zip(std_m, 
                                                                 inverse_X_names,
                                                                 np.mean(inverse_X, axis=0),
                                                                 cluster_X_mean,
                                                                 above_mean,
                                                                 np.std(inverse_X, axis=0))], 
                                                                   reverse=True)
                above_stats = [a for a in stats if a[4] == True]
                below_stats = [a for a in stats if a[4] == False]
                
                if len(above_stats) > 0:
                    cluster_stats += "<h3>Above Average</h3><table><tr><th>Feature</th>" \
                                     + "<th style='width:50px;'><small>Mean</small></th>" \
                                     + "<th style='width:50px'><small>STD</small></th></tr>"
                    for s,f,i,c,a,v in above_stats[:5]:
                        cluster_stats += "<tr><td>%s</td><td><small>%s</small></td>"%(f, round(c,3)) \
                                       + "<td class='std'><small>%sx</small></td></tr>"%(round(s,1))
                    cluster_stats += "</table>"
                if len(below_stats) > 0:
                    cluster_stats += "<h3>Below Average</h3><table><tr><th>Feature</th>" \
                                     + "<th style='width:50px;'><small>Mean</small></th>" \
                                     + "<th style='width:50px'><small>STD</small></th></tr>"
                    for s,f,i,c,a,v in below_stats[:5]:
                        cluster_stats += "<tr><td>%s</td><td><small>%s</small></td>"%(f, round(c,3)) \
                                       + "<td class='std'><small>%sx</small></td></tr>"%(round(s,1))
                    cluster_stats += "</table>"
            cluster_stats += "<h3>Size</h3><p>%s</p>"%(len(member_ids))
            return "%s"%(str(cluster_stats))
            
        def _format_projection_statistics(member_ids, projected_X, projected_X_names):
            projection_stats = ""
            if projected_X is not None:
                projection_stats += "<h3>Projection</h3><table><tr><th>Lens</th><th style='width:50px;'>" \
                                    + "<small>Mean</small></th><th style='width:50px;'><small>Max</small></th>" \
                                    + "<th style='width:50px;'><small>Min</small></th></tr>"
                if isinstance(projected_X_names, list):
                    projected_X_names = np.array(projected_X_names)
                # Create defaults when providing no projected_X_names
                if projected_X_names.shape[0] == 0:
                    projected_X_names = np.array(["p_%s"%(i) for i in range(projected_X.shape[1])])

                means_v = np.mean(projected_X[member_ids], axis=0)
                maxs_v = np.max(projected_X[member_ids], axis=0)
                mins_v = np.min(projected_X[member_ids], axis=0)

                for name, mean_v, max_v, min_v in zip(projected_X_names, 
                                                      means_v, 
                                                      maxs_v, 
                                                      mins_v):
                    projection_stats += "<tr><td>%s</td><td><small>%s</small></td><td><small>%s</small>"%(name, 
                                                                                                          round(mean_v, 3), 
                                                                                                          round(max_v, 3)) \
                                      + "</td><td><small>%s</small></td></tr>"%(round(min_v, 3))
                projection_stats += "</table>"
            return projection_stats

        def _format_tooltip(member_ids, custom_tooltips, inverse_X, 
                            inverse_X_names, projected_X, projected_X_names):
            
            tooltip = _format_projection_statistics(member_ids, projected_X, projected_X_names)
            tooltip += _format_cluster_statistics(member_ids, inverse_X, inverse_X_names)

            if custom_tooltips is not None:
                tooltip += "<h3>Members</h3>"
                for custom_tooltip in custom_tooltips[member_ids]:
                    tooltip += "%s "%(custom_tooltip)
            return tooltip

        def _format_meta(graph, custom_meta):
            meta = ""
            if custom_meta is not None:
                for k, v in custom_meta:
                    meta += "<h3>%s</h3>\n<p>%s</p>\n"%(k, v)
            meta += "<h3>Nodes</h3><p>%s</p>"%(len(graph["nodes"]))
            meta += "<h3>Edges</h3><p>%s</p>"%(sum([len(l) for l in graph["links"].values()]))
            meta += "<h3>Total Samples</h3><p>%s</p>"%(sum([len(l) for l in graph["nodes"].values()]))
            n = [l for l in graph["nodes"].values()]
            n_unique = len(set([i for s in n for i in s]))
            meta += "<h3>Unique Samples</h3><p>%s</p>"%(n_unique)
            return meta

        def _color_function(member_ids, color_function):
            return int(np.mean(color_function[member_ids]) * 30)

        def _size_node(member_ids):
            return int(np.log(len(member_ids) + 1) + 1)

        def _type_node():
            return "circle"

        def _size_link_width(graph, node_id, linked_node_id):
            return 1

        def _dict_to_json(graph, color_function, inverse_X, 
                          inverse_X_names, projected_X, projected_X_names):
            json_dict = {"nodes": [], "links": []}
            node_id_to_num = {}
            for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
                node_id_to_num[node_id] = i
                n = { "id": "",
                      "name": "node_id",
                      "color": _color_function(member_ids, color_function),
                      "type": _type_node(),
                      "size": _size_node(member_ids),
                      "tooltip": _format_tooltip(member_ids, 
                                                 custom_tooltips, 
                                                 inverse_X, 
                                                 inverse_X_names, 
                                                 projected_X, 
                                                 projected_X_names)}
                json_dict["nodes"].append(n)
            for i, (node_id, linked_node_ids) in enumerate(graph["links"].items()):
                for linked_node_id in linked_node_ids:
                    l = { "source": node_id_to_num[node_id], 
                         "target": node_id_to_num[linked_node_id],
                         "width": _size_link_width(graph, node_id, linked_node_id)}
                    json_dict["links"].append(l)
            return json.dumps(json_dict)

        def _color_function_distribution(graph, color_function):
            bin_colors = { 0: "#FF2800",
                           1: "#FF6400",
                           2: "#FFa000",
                           3: "#FFdc00",
                           4: "#b0ff00",
                           5: "#00ff36",
                           6: "#00e4ff",
                           7: "#0084ff",
                           8: "#0022ff",
                           9: "#0300ff" }

            dist = '  <h3>Distribution</h3>\n  <div id="histogram">\n'
            buckets = defaultdict(float)
            
            for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
                # round to color range value to nearest 3 multiple
                k = int(round(_color_function(member_ids, color_function) / 3.0))
                buckets[k] += len(member_ids)

            # TODO: Fix color-range length of 31 (prob not the best idea to pick
            #       prime numbers for equidistant binning...)
            buckets[9] += buckets[10]

            max_bucket_value = max(buckets.values())
            sum_bucket_value = sum(list(set(buckets.values())))
            for bucket_id in range(10):
                bucket_value = buckets[bucket_id]
                height = int(((bucket_value / max_bucket_value) * 100) + 5)
                perc = round((bucket_value / sum_bucket_value) * 100.,1)
                dist += '    <div class="bin" style="height: %spx; background:%s">\n'%(height,
                                                                                       bin_colors[bucket_id]) + \
                        '      <div>%s%%</div>\n'%(perc) + \
                        '    </div>\n'
            dist += '  </div>'
            return dist

        color_function = _init_color_function(graph, color_function)
        json_graph = _dict_to_json(graph, color_function, inverse_X, inverse_X_names, projected_X, projected_X_names)
        color_distribution = _color_function_distribution(graph, color_function)
        meta = _format_meta(graph, custom_meta)

        template = """<!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="generator" content="KeplerMapper">
        <title>%s | KeplerMapper</title>
        <link rel="icon" type="image/png" href="https://i.imgur.com/2eZcTZn.png" />
        <link href='https://fonts.googleapis.com/css?family=Roboto+Mono:700,300' 
              rel='stylesheet' type='text/css'>
        <style>
          * {margin: 0; padding: 0;}
          html, body {height: 100%%;}
          body {font-family: "Roboto Mono", "Helvetica", sans-serif; font-size:14px;}
          #display {color: #95A5A6; background: #212121;}
          #print {color: #000; background: #FFF;}
          h1 {font-size: 21px; font-weight: 300; font-weight: 300;}
          h2 {font-size: 18px; padding-bottom: 20px; font-weight: 300;}
          h3 {font-size: 14px; font-weight: 700; text-transform: uppercase;}
          #meta h3 { float: left; padding-right: 8px;}
          p, #tooltip h3, ol, ul, table {padding-bottom: 20px;}
          ol, ul {padding-left: 20px;}
          ol b {display: block;}
          a {color: #16a085; text-decoration: none;}
          a:hover {color: #2ecc71;}
          #header {height: 35px; padding: 20px; position: absolute; top: 0; left: 0; right: 0; 
                   z-index: 9999;}
          #display #header {background: #111111; box-shadow: 0px 0px 4px #000}
          #print #header {background: #FFF;}
          #canvas {height: 100%%; width: 100%%; display: block;}
          #tooltip {position: absolute; top: 75px; left: 0; bottom: 0;  
                    width: 320px; padding: 20px; overflow: auto; display: none;}
          #display #tooltip {background: #191919;}
          #print #tooltip {background: #FFF;}
          #meta {position: absolute; top: 75px; right: 0; bottom: 0; 
                 width: 320px; padding: 20px; overflow: auto;}
          #display #meta {background: #191919;}
          #print #meta {background: #FFF; }
          #meta_control, #tooltip_control {position: absolute; right: 20px;}
          #meta::-webkit-scrollbar, #tooltip::-webkit-scrollbar {width: 1em;}
          #meta::-webkit-scrollbar-track, #tooltip::-webkit-scrollbar-track {
            -webkit-box-shadow: inset 0 0 6px rgba(0,0,0,0.3);}
          #meta::-webkit-scrollbar-thumb, #tooltip::-webkit-scrollbar-thumb {
            background-color: darkgrey; outline: 1px solid slategrey;}
          #histogram { display: block; height: 100px; padding-top: 50px; clear: both;}
          #display #histogram {opacity: 0.68;}
          .bin {width: 30px; float: left;}
          .bin div { font-size: 10px; display: block; width: 35px; margin-top: -30px; 
                     text-align: right; margin-left:-3px;
                     -webkit-transform: rotate(-90deg); -moz-transform: rotate(-90deg); 
                     -ms-transform: rotate(-90deg); -o-transform: rotate(-90deg);}
          #histogram:hover {opacity:1.;}
          #display .circle {stroke-opacity:0.18; stroke-width: 7px; stroke: #000;}
          #print .circle {stroke-opacity:1; stroke-width: 2px; stroke: #000; 
                          stroke-linecap: round;}
          #print .link {stroke: #000;}
          #display .link {stroke: rgba(160,160,160, 0.5);}
          table { border-collapse: collapse; display: table; width: 100%%; margin-bottom:20px;}
          td, th { padding: 5px; text-align: left;}
          #display th { background: #212121}
          td { border-bottom: 1px solid #111;}
        </style>
      </head>
      <body id="display">
        <div id="header">
          <noscript><b>Requires JavaScript (d3.js) for visualizations</b></noscript>
          <h1>%s</h1>
        </div>
        <div id="canvas">

        </div>
        <div id="tooltip">
          <div id="tooltip_control">
            <a href="#"><small>[-]</small></a>
          </div>
          <h2>Cluster Meta</h2>
          <div id="tooltip_content">
          
          </div>
        </div>
        <div id="meta">
          <div id="meta_control">
            <a href="#"><small>[-]</small></a>
          </div>
          <h2>Graph Meta</h2>
          %s
          %s
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
        <script>
        // Height and width settings
        var canvas_height = window.innerHeight - 5;
        document.getElementById("canvas").style.height = canvas_height + "px";          
        var width = document.getElementById("canvas").offsetWidth;
        var height = document.getElementById("canvas").offsetHeight;
        var w = width;
        var h = height;
        
        // We draw the graph in SVG
        var svg = d3.select("#canvas").append("svg")
                  .attr("width", width)
                  .attr("height", height);

        var focus_node = null, highlight_node = null;
        var text_center = false;
        var outline = false;

        // Size for zooming
        var size = d3.scale.pow().exponent(1)
                   .domain([1,100])
                   .range([8,24]);

        // Show/Hide Functionality
        d3.select("#tooltip_control").on("click", function() {
          d3.select("#tooltip").style("display", "none");
        })
        d3.select("#meta_control").on("click", function() {
          d3.select("#meta").style("display", "none");
        })

        // Color settings: Ordinal Scale of ["0"-"30"] hot-to-cold
        var color = d3.scale.ordinal() 
                    .domain(["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "10", 
                             "11", "12", "13","14","15","16","17","18","19","20",
                             "21","22","23","24","25","26","27","28","29","30"])
                    .range(["#FF0000","#FF1400","#FF2800","#FF3c00","#FF5000","#FF6400",
                            "#FF7800","#FF8c00","#FFa000","#FFb400","#FFc800","#FFdc00",
                            "#FFf000","#fdff00","#b0ff00","#65ff00","#17ff00","#00ff36",
                            "#00ff83","#00ffd0","#00e4ff","#00c4ff","#00a4ff","#00a4ff",
                            "#0084ff","#0064ff","#0044ff","#0022ff","#0002ff","#0100ff",
                            "#0300ff","#0500ff"]);
        // Force settings
        var force = d3.layout.force()
                    .linkDistance(5)
                    .gravity(0.2)
                    .charge(-1200)
                    .size([w,h]);

        // Variety of variable inits
        var highlight_color = "blue";
        var highlight_trans = 0.1;        
        var default_node_color = "#ccc";
        var default_node_color = "rgba(160,160,160, 0.5)";
        var default_link_color = "rgba(160,160,160, 0.5)";
        var nominal_base_node_size = 8;
        var nominal_text_size = 15;
        var max_text_size = 24;
        var nominal_stroke = 1.;
        var max_stroke = 4.5;
        var max_base_node_size = 36;
        var min_zoom = 0.1;
        var max_zoom = 7;
        var zoom = d3.behavior.zoom().scaleExtent([min_zoom,max_zoom])
        var g = svg.append("g");
        
        svg.style("cursor","move");

        graph = %s;
            
        force
          .nodes(graph.nodes)
          .links(graph.links)
          .start();

        var link = g.selectAll(".link")
                    .data(graph.links)
                    .enter().append("line")
                    .attr("class", "link")
                    .style("stroke-width", function(d) { return d.w * nominal_stroke; })
                    .style("stroke-width", function(d) { return d.w * nominal_stroke; })
                    //.style("stroke", function(d) { 
                    //  if (isNumber(d.score) && d.score>=0) return color(d.score);
                    //  else return default_link_color; })

        var node = g.selectAll(".node")
                    .data(graph.nodes)
                    .enter().append("g")
                    .attr("class", "node")
                    .call(force.drag)

        node.on("dblclick.zoom", function(d) { d3.event.stopPropagation();
          var dcx = (window.innerWidth/2-d.x*zoom.scale());
          var dcy = (window.innerHeight/2-d.y*zoom.scale());
          zoom.translate([dcx,dcy]);
          g.attr("transform", "translate("+ dcx + "," + dcy  + ")scale(" + zoom.scale() + ")");
        });

        var tocolor = "fill";
        var towhite = "stroke";
        if (outline) {
          tocolor = "stroke"
          towhite = "fill"
        }

        // Drop-shadow Filter
        var svg = d3.select("svg");
        var defs = svg.append("defs");
        var dropShadowFilter = defs.append('svg:filter')
          .attr('id', 'drop-shadow')
          .attr('filterUnits', "userSpaceOnUse")
          .attr('width', '250%%')
          .attr('height', '250%%');
        dropShadowFilter.append('svg:feGaussianBlur')
          .attr('in', 'SourceGraphic')
          .attr('stdDeviation', 12)
          .attr('result', 'blur-out');
        dropShadowFilter.append('svg:feColorMatrix')
          .attr('in', 'blur-out')
          .attr('type', 'hueRotate')
          .attr('values', 0)
          .attr('result', 'color-out');
        dropShadowFilter.append('svg:feOffset')
          .attr('in', 'color-out')
          .attr('dx', 0)
          .attr('dy', 0)
          .attr('result', 'the-shadow');
        dropShadowFilter.append('svg:feComponentTransfer')
          .attr('type', 'linear')
          .attr('slope', 0.2)
          .attr('result', 'shadow-opacity');
        dropShadowFilter.append('svg:feBlend')
          .attr('in', 'SourceGraphic')
          .attr('in2', 'the-shadow')
          .attr('mode', 'normal');

      var circle = node.append("path")
        .attr("d", d3.svg.symbol()
        .size(function(d) { return d.size * 50; })
        .type(function(d) { return d.type; }))
        .attr("class", "circle")
        .style(tocolor, function(d) { 
          return color(d.color);
        })
        //.style("filter", "url(#drop-shadow)");

      var text = g.selectAll(".text")
        .data(graph.nodes)
        .enter().append("text")
        .attr("dy", ".35em")
        .style("font-family", "Roboto")
        .style("font-weight", "400")
        .style("color", "#2C3E50")
        .style("font-size", nominal_text_size + "px")

      if (text_center)
        text.text(function(d) { return d.id; })
        .style("text-anchor", "middle");
      else 
        text.attr("dx", function(d) {return (size(d.size)||nominal_base_node_size);})
        .text(function(d) { return '\u2002'+d.id; });
      
      // Mouse events
      node.on("mouseover", function(d) {
        set_highlight(d);
        console.log("node hober");

        d3.select("#tooltip").style("display", "block");
        d3.select("#tooltip_content").html(d.tooltip + "<br/>");
        }).on("mousedown", function(d) { 
        d3.event.stopPropagation();
        focus_node = d;
        if (highlight_node === null) set_highlight(d)
      }).on("mouseout", function(d) {
        console.log("mouseout");
        exit_highlight();
      });

      d3.select(window).on("mouseup", function() {
        if (focus_node!==null){
          focus_node = null;
        }
        if (highlight_node === null) exit_highlight();
      });

      // Node highlighting logic
      function exit_highlight(){
        highlight_node = null;
        if (focus_node===null){
          svg.style("cursor","move"); 
        }
      }

      function set_highlight(d){
        svg.style("cursor","pointer");
        if (focus_node!==null) d = focus_node;
      }

      // Zoom logic
      zoom.on("zoom", function() {
        var stroke = nominal_stroke;
        var base_radius = nominal_base_node_size;
        if (nominal_base_node_size*zoom.scale()>max_base_node_size) {
          base_radius = max_base_node_size/zoom.scale();}
        circle.attr("d", d3.svg.symbol()
          .size(function(d) { return d.size * 50; })
          .type(function(d) { return d.type; }))
        if (!text_center) text.attr("dx", function(d) { 
          return (size(d.size)*base_radius/nominal_base_node_size||base_radius); });
                
        var text_size = nominal_text_size;
        if (nominal_text_size*zoom.scale()>max_text_size) {
          text_size = max_text_size/zoom.scale(); }
        text.style("font-size",text_size + "px");

        g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
      });

      svg.call(zoom);   
      resize();
      d3.select(window).on("resize", resize);

      // Animation per tick
      force.on("tick", function() {
        node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
        text.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
        link.attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });
        node.attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
      });

      // Resizing window and redraws
      function resize() {
        var width = window.innerWidth, height = window.innerHeight;
        var width = document.getElementById("canvas").offsetWidth;
        var height = document.getElementById("canvas").offsetHeight;
        svg.attr("width", width).attr("height", height);
        
        force.size([force.size()[0]+(width-w)/zoom.scale(),
                    force.size()[1]+(height-h)/zoom.scale()]).resume();
        w = width;
        h = height;
      }

      function isNumber(n) {
        return !isNaN(parseFloat(n)) && isFinite(n);
      }

      // Key press events
      window.addEventListener("keydown", function (event) {
      if (event.defaultPrevented) {
        return; // Do nothing if the event was already processed
      }
        switch (event.key) {
          case "s":
            // Do something for "s" key press.
            node.style("filter", "url(#drop-shadow)");
            break;
          case "c":
            // Do something for "s" key press.
            node.style("filter", null);
            break;
          case "p":
            // Do something for "p" key press.
            d3.select("body").attr('id', null).attr('id', "print")
            break;
          case "d":
            // Do something for "d" key press.
            d3.select("body").attr('id', null).attr('id', "display")
            break;
          case "z":
            force.gravity(0.)
                 .charge(0.);
            resize();
            break
          case "m":
            force.gravity(0.07)
                 .charge(-1);
            resize();
            break
          case "e":
            force.gravity(0.4)
                 .charge(-600);
              
            resize();
            break
          default:
            return; // Quit when this doesn't handle the key event.
        }
        // Cancel the default action to avoid it being handled twice
        event.preventDefault();
      }, true);
      </script>
      </body>
    </html>"""%(title,
                title,
                meta, 
                color_distribution, 
                json_graph)
        if save_file:
            with open(path_html, "wb") as outfile:
                if self.verbose > 0:
                    print("Wrote visualization to: %s"%(path_html))
                outfile.write(template.encode("utf-8"))
        return template

    def data_from_cluster_id(self, cluster_id, graph, data):
        """Returns the original data of each cluster member for a given cluster ID

        Input: cluster_id. Integer. ID of the cluster.
               graph. Dict. The resulting dictionary after applying map()
               data. Numpy array. Original dataset. Accepts both 1-D and 2-D array.
        Output: rows of cluster member data as Numpy array.
        """
        if cluster_id in graph["nodes"]:
            cluster_members = graph["nodes"][cluster_id]
            cluster_members_data = data[cluster_members]
            return cluster_members_data
        else:
            return np.array([])
