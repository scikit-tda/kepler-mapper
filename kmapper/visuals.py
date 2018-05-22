# A small helper class to house functions needed by KeplerMapper.visualize
import numpy as np
from sklearn import preprocessing
import json
from collections import defaultdict


palette = [
    '#0500ff', '#0300ff', '#0100ff', '#0002ff', '#0022ff', '#0044ff',
    '#0064ff', '#0084ff', '#00a4ff', '#00a4ff', '#00c4ff', '#00e4ff',
    '#00ffd0', '#00ff83', '#00ff36', '#17ff00', '#65ff00', '#b0ff00',
    '#fdff00', '#FFf000', '#FFdc00', '#FFc800', '#FFb400', '#FFa000',
    '#FF8c00', '#FF7800', '#FF6400', '#FF5000', '#FF3c00', '#FF2800',
    '#FF1400', '#FF0000'
]


def init_color_function(graph, color_function=None):
    # If no color_function provided we color by row order in data set
    # Reshaping to 2-D array is required for sklearn 0.19
    n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
    if color_function is None:
        color_function = np.arange(n_samples).reshape(-1, 1)
    else:
        color_function = color_function.reshape(-1, 1)

    color_function = color_function.astype(np.float64)
    # MinMax Scaling to be friendly to non-scaled input.
    scaler = preprocessing.MinMaxScaler()
    color_function = scaler.fit_transform(color_function).ravel()
    return color_function


def format_meta(graph, custom_meta=None):

    n = [l for l in graph["nodes"].values()]
    n_unique = len(set([i for s in n for i in s]))

    if custom_meta is None:
        custom_meta = graph['meta_data']

    mapper_summary = {
        "custom_meta": custom_meta,
        "n_nodes": len(graph["nodes"]),
        "n_edges": sum([len(l) for l in graph["links"].values()]),
        "n_total": sum([len(l) for l in graph["nodes"].values()]),
        "n_unique": n_unique
    }

    return mapper_summary


def format_mapper_data(graph, color_function, X,
                       X_names, lens, lens_names, custom_tooltips, env):
    # import pdb; pdb.set_trace()
    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
        node_id_to_num[node_id] = i
        c = _color_function(member_ids, color_function)
        t = _type_node()
        s = _size_node(member_ids)
        tt = _format_tooltip(env, member_ids, custom_tooltips, X, X_names, lens, lens_names, color_function)

        n = {"id": "",
             "name": node_id,
             "color": c,
             "type": _type_node(),
             "size": s,
             "tooltip": tt}

        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in enumerate(graph["links"].items()):
        for linked_node_id in linked_node_ids:
            l = {"source": node_id_to_num[node_id],
                 "target": node_id_to_num[linked_node_id],
                 "width": _size_link_width(graph, node_id, linked_node_id)}
            json_dict["links"].append(l)
    return json_dict


def build_histogram(data):
    # Build histogram of data based on values of color_function

    h_min, h_max = 0, 1
    hist, bin_edges = np.histogram(data, range=(h_min, h_max), bins=10)

    bin_mids = np.mean(np.array(list(zip(bin_edges, bin_edges[1:]))), axis=1)

    histogram = []
    max_bucket_value = max(hist)
    sum_bucket_value = sum(hist)
    for bar, mid in zip(hist, bin_mids):
        height = int(((bar / max_bucket_value) * 100) + 1)
        perc = round((bar / sum_bucket_value) * 100., 1)
        color = palette[_color_idx(mid)]

        histogram.append({
            'height': height,
            'perc': perc,
            'color': color
        })
    return histogram


def graph_data_distribution(graph, color_function):

    node_averages = []
    for node_id, member_ids in graph["nodes"].items():
        member_colors = color_function[member_ids]
        node_averages.append(np.mean(member_colors))

    histogram = build_histogram(node_averages)

    return histogram


def _format_cluster_statistics(member_ids, X, X_names):
    # TODO: Cache X_mean and X_std for all clusters.
    # TODO: replace long tuples with named tuples.
    # TODO: Name all the single letter variables.
    # TODO: remove duplication between above_stats and below_stats
    # TODO: Should we only show variables that are much above or below the mean?

    cluster_data = {'above': [], 'below': [], 'size': len(member_ids)}

    cluster_stats = ""
    if X is not None:
        # List vs. numpy handling: cast to numpy array
        if isinstance(X_names, list):
            X_names = np.array(X_names)
        # Defaults when providing no X_names
        if X_names.shape[0] == 0:
            X_names = np.array(["f_%s" % (i) for i in range(
                X.shape[1])])

        cluster_X_mean = np.mean(X[member_ids], axis=0)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        above_mean = cluster_X_mean > X_mean
        std_m = np.sqrt((cluster_X_mean - X_mean)**2) / X_std

        stat_zip = list(zip(std_m, X_names, np.mean(X, axis=0), cluster_X_mean, above_mean, np.std(X, axis=0)))
        stats = sorted(stat_zip, reverse=True)
        above_stats = [a for a in stats if a[4] == True]
        below_stats = [a for a in stats if a[4] == False]

        if len(above_stats) > 0:
            for s, f, i, c, a, v in above_stats[:5]:
                cluster_data['above'].append({
                    'feature': f,
                    'mean': round(c, 3),
                    'std': round(s, 1)
                })

        if len(below_stats) > 0:
            for s, f, i, c, a, v in below_stats[:5]:
                cluster_data['below'].append({
                    'feature': f,
                    'mean': round(c, 3),
                    'std': round(s, 1)
                })

    return cluster_data


def _format_projection_statistics(member_ids, lens, lens_names):
    projection_data = []

    if lens is not None:
        if isinstance(lens_names, list):
            lens_names = np.array(lens_names)

        # Create defaults when providing no lens_names
        if lens_names.shape[0] == 0:
            lens_names = np.array(
                ["p_%s" % (i) for i in range(lens.shape[1])])

        means_v = np.mean(lens[member_ids], axis=0)
        maxs_v = np.max(lens[member_ids], axis=0)
        mins_v = np.min(lens[member_ids], axis=0)

        for name, mean_v, max_v, min_v in zip(lens_names, means_v, maxs_v, mins_v):
            projection_data.append({
                'name': name,
                'mean': round(mean_v, 3),
                'max': round(max_v, 3),
                'min': round(min_v, 3)
            })

    return projection_data


def _format_tooltip(env, member_ids, custom_tooltips, X,
                    X_names, lens, lens_names, color_function):
    # TODO: Allow customization in the form of aggregate per node and per entry in node.
    # TODO: Allow users to turn off tooltip completely.

    custom_tooltips = custom_tooltips[member_ids] if custom_tooltips is not None else member_ids

    # list will render better than numpy arrays
    custom_tooltips = list(custom_tooltips)

    projection_stats = _format_projection_statistics(
        member_ids, lens, lens_names)
    cluster_stats = _format_cluster_statistics(member_ids, X, X_names)

    histogram = build_histogram(color_function[member_ids])

    tooltip = env.get_template('cluster_tooltip.html').render(
        projection_stats=projection_stats,
        cluster_stats=cluster_stats,
        custom_tooltips=custom_tooltips,
        histogram=histogram,
        dist_label="Member")

    return tooltip


def _color_function(member_ids, color_function):
    return _color_idx(np.mean(color_function[member_ids]))
    # return int(np.mean(color_function[member_ids]) * 30)


def _color_idx(val):
    """ Take a value between 0 and 1 and return the idx of color """
    return int(val * 30)


def _size_node(member_ids):
    return int(np.log(len(member_ids) + 1) + 1)


def _type_node():
    return "circle"


def _size_link_width(graph, node_id, linked_node_id):
    return 1
