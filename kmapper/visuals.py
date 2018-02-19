# A small helper class to house functions needed by KeplerMapper.visualize
import numpy as np
from sklearn import preprocessing
import json
from collections import defaultdict


def init_color_function(graph, color_function=None):
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


def format_meta(graph, custom_meta):
    meta = ""
    if custom_meta is not None:
        for k, v in custom_meta:
            meta += "<h3>%s</h3>\n<p>%s</p>\n" % (k, v)
    meta += "<h3>Nodes</h3><p>%s</p>" % (len(graph["nodes"]))
    meta += "<h3>Edges</h3><p>%s</p>" % (sum([len(l)
                                              for l in graph["links"].values()]))
    meta += "<h3>Total Samples</h3><p>%s</p>" % (
        sum([len(l) for l in graph["nodes"].values()]))
    n = [l for l in graph["nodes"].values()]
    n_unique = len(set([i for s in n for i in s]))
    meta += "<h3>Unique Samples</h3><p>%s</p>" % (n_unique)
    return meta


def dict_to_json(graph, color_function, inverse_X,
                 inverse_X_names, projected_X, projected_X_names, custom_tooltips):
    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
        node_id_to_num[node_id] = i
        n = {"id": "",
             "name": node_id,
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
            l = {"source": node_id_to_num[node_id],
                 "target": node_id_to_num[linked_node_id],
                 "width": _size_link_width(graph, node_id, linked_node_id)}
            json_dict["links"].append(l)
    return json.dumps(json_dict)


def color_function_distribution(graph, color_function):
    bin_colors = {0: "#FF2800",
                  1: "#FF6400",
                  2: "#FFa000",
                  3: "#FFdc00",
                  4: "#b0ff00",
                  5: "#00ff36",
                  6: "#00e4ff",
                  7: "#0084ff",
                  8: "#0022ff",
                  9: "#0300ff"}

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
        perc = round((bucket_value / sum_bucket_value) * 100., 1)
        dist += '    <div class="bin" style="height: %spx; background:%s">\n' % (height,
                                                                                 bin_colors[bucket_id]) + \
                '      <div>%s%%</div>\n' % (perc) + \
                '    </div>\n'
    dist += '  </div>'
    return dist


def _format_cluster_statistics(member_ids, inverse_X, inverse_X_names):
    cluster_stats = ""
    if inverse_X is not None:
        # List vs. numpy handling: cast to numpy array
        if isinstance(inverse_X_names, list):
            inverse_X_names = np.array(inverse_X_names)
        # Defaults when providing no inverse_X_names
        if inverse_X_names.shape[0] == 0:
            inverse_X_names = np.array(["f_%s" % (i) for i in range(
                inverse_X.shape[1])])

        cluster_X_mean = np.mean(inverse_X[member_ids], axis=0)
        inverse_X_mean = np.mean(inverse_X, axis=0)
        inverse_X_std = np.std(inverse_X, axis=0)
        above_mean = cluster_X_mean > inverse_X_mean
        std_m = np.sqrt((cluster_X_mean - inverse_X_mean)**2) / inverse_X_std

        stats = sorted([(s, f, i, c, a, v) for s, f, i, c, a, v in zip(std_m,
                                                                       inverse_X_names,
                                                                       np.mean(
                                                                           inverse_X, axis=0),
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
            for s, f, i, c, a, v in above_stats[:5]:
                cluster_stats += "<tr><td>%s</td><td><small>%s</small></td>" % (f, round(c, 3)) \
                    + "<td class='std'><small>%sx</small></td></tr>" % (round(s, 1))
            cluster_stats += "</table>"
        if len(below_stats) > 0:
            cluster_stats += "<h3>Below Average</h3><table><tr><th>Feature</th>" \
                             + "<th style='width:50px;'><small>Mean</small></th>" \
                             + "<th style='width:50px'><small>STD</small></th></tr>"
            for s, f, i, c, a, v in below_stats[:5]:
                cluster_stats += "<tr><td>%s</td><td><small>%s</small></td>" % (f, round(c, 3)) \
                    + "<td class='std'><small>%sx</small></td></tr>" % (round(s, 1))
            cluster_stats += "</table>"
    cluster_stats += "<h3>Size</h3><p>%s</p>" % (len(member_ids))
    return "%s" % (str(cluster_stats))


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
            projected_X_names = np.array(
                ["p_%s" % (i) for i in range(projected_X.shape[1])])

        means_v = np.mean(projected_X[member_ids], axis=0)
        maxs_v = np.max(projected_X[member_ids], axis=0)
        mins_v = np.min(projected_X[member_ids], axis=0)

        for name, mean_v, max_v, min_v in zip(projected_X_names,
                                              means_v,
                                              maxs_v,
                                              mins_v):
            projection_stats += "<tr><td>%s</td><td><small>%s</small></td><td><small>%s</small>" % (name,
                                                                                                    round(
                                                                                                        mean_v, 3),
                                                                                                    round(max_v, 3)) \
                + "</td><td><small>%s</small></td></tr>" % (round(min_v, 3))
        projection_stats += "</table>"
    return projection_stats


def _format_tooltip(member_ids, custom_tooltips, inverse_X,
                    inverse_X_names, projected_X, projected_X_names):

    tooltip = _format_projection_statistics(
        member_ids, projected_X, projected_X_names)
    tooltip += _format_cluster_statistics(member_ids,
                                          inverse_X, inverse_X_names)

    if custom_tooltips is not None:
        tooltip += "<h3>Members</h3>"
        for custom_tooltip in custom_tooltips[member_ids]:
            tooltip += "%s " % (custom_tooltip)
    return tooltip


def _color_function(member_ids, color_function):
    return int(np.mean(color_function[member_ids]) * 30)


def _size_node(member_ids):
    return int(np.log(len(member_ids) + 1) + 1)


def _type_node():
    return "circle"


def _size_link_width(graph, node_id, linked_node_id):
    return 1
