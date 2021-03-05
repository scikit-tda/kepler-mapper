# A small helper class to house functions needed by KeplerMapper.visualize
import numpy as np
import scipy.sparse
from sklearn import preprocessing
import json
from collections import defaultdict
from ast import literal_eval
from .utils import deprecated_alias
import os
from jinja2 import Environment, FileSystemLoader, Template, StrictUndefined

colorscale_default = [
    [0.0, "rgb(68, 1, 84)"],  # Viridis
    [0.1, "rgb(72, 35, 116)"],
    [0.2, "rgb(64, 67, 135)"],
    [0.3, "rgb(52, 94, 141)"],
    [0.4, "rgb(41, 120, 142)"],
    [0.5, "rgb(32, 144, 140)"],
    [0.6, "rgb(34, 167, 132)"],
    [0.7, "rgb(68, 190, 112)"],
    [0.8, "rgb(121, 209, 81)"],
    [0.9, "rgb(189, 222, 38)"],
    [1.0, "rgb(253, 231, 36)"],
]


palette = [
    "#0500ff",
    "#0300ff",
    "#0100ff",
    "#0002ff",
    "#0022ff",
    "#0044ff",
    "#0064ff",
    "#0084ff",
    "#00a4ff",
    "#00a4ff",
    "#00c4ff",
    "#00e4ff",
    "#00ffd0",
    "#00ff83",
    "#00ff36",
    "#17ff00",
    "#65ff00",
    "#b0ff00",
    "#fdff00",
    "#FFf000",
    "#FFdc00",
    "#FFc800",
    "#FFb400",
    "#FFa000",
    "#FF8c00",
    "#FF7800",
    "#FF6400",
    "#FF5000",
    "#FF3c00",
    "#FF2800",
    "#FF1400",
    "#FF0000",
]


def colorscale_from_matplotlib_cmap(cmap, ii_off=0, ff_off=0, nbins=10):
    """Create a colorscale from a matplotlib colormap.

    See https://matplotlib.org/tutorials/colors/colormaps.html
    for more details about matplotlib colormaps.

    Parameters
    ----------

    cmap : matplotlib.colors.LinearSegmentedColormap
        A matplotlib colormap

    ii_off : int
        The starting index offset to use when sampling the matplotlib
        colormap. Must be in the range 0-255.

    ff_off : int
        The ending index offset to use when sampling the matplotlib
        colormap. Must be in the range 0-255.

    nbins : int
        Number of bins (i.e. samples of the colormap) to take when
        constructing the colorscale.

    Returns
    -------

    colorscale
        A colorscale

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> # use a non-truncated colormap
    >>> colorscale = colorscale_from_matplotlib_cmap(plt.cm.cool)

    >>> import matplotlib.pyplot as plt
    >>> # skip the first 10% of the matplotlib colormap
    >>> colorscale = colorscale_from_matplotlib_cmap(plt.cm.cool, ii_off=255//10)

    >>> import matplotlib.pyplot as plt
    >>> # skip the last 10% of the matplotlib colormap
    >>> colorscale = colorscale_from_matplotlib_cmap(plt.cm.cool, ff_off=255//10)

    """
    if cmap.N != 256:
        raise ValueError("Not implemented for colormaps with cmap.N != 256")

    if ii_off + ff_off > 256:
        raise ValueError("ii_off + ff_off must be less than 256")

    ii = 0 + ii_off
    ff = cmap.N - ff_off
    sk = (cmap.N - ii_off - ff_off) // (nbins + 1)
    cmap_list = [
        cmap(el) for el in np.arange(cmap.N)[ii:ff:sk]
    ]
    rgb_strings = [
        "rgb({}, {}, {})".format(
            int(255 * el[0]), int(255 * el[1]), int(255 * el[2])
        ) for el in cmap_list
    ]
    if len(cmap_list) != nbins + 1:
        raise ValueError("Failed to build correct size colorscale")

    return list(zip(np.arange(nbins + 1) / nbins, rgb_strings))


def _colors_to_rgb(colorscale):
    """ Ensure that the color scale is formatted in rgb strings.
        If the colorscale is a hex string, then convert to rgb.
    """
    if colorscale[0][1][0] == "#":
        plotly_colors = np.array(colorscale)[:, 1].tolist()
        for k, hexcode in enumerate(plotly_colors):
            hexcode = hexcode.lstrip("#")
            hex_len = len(hexcode)
            step = hex_len // 3
            colorscale[k][1] = "rgb" + str(
                tuple(int(hexcode[j : j + step], 16) for j in range(0, hex_len, step))
            )

    return colorscale


def _to_html_format(st):
    return st.replace("\n", "<br>")


def _map_val2color(val, vmin, vmax, colorscale=None):
    """ Maps a value val in [vmin, vmax] to the corresponding color in
        the colorscale
        returns the rgb color code of that color
    """
    colorscale = colorscale or colorscale_default

    if vmin >= vmax:
        raise ValueError("vmin should be < vmax")

    scale = list(map(float, np.array(colorscale)[:, 0]))
    colors = np.array(colorscale)[:, 1]

    colors_01 = (
        np.array(list(map(literal_eval, [color[3:] for color in colors]))) / 255.0
    )

    v = (val - vmin) / float((vmax - vmin))  # val is mapped to v in[0,1]

    idx = 0
    # sequential search for the two   consecutive indices idx, idx+1 such that
    # v belongs to the interval  [scale[idx], scale[idx+1]
    while v > scale[idx + 1]:
        idx += 1
    left_scale_val = scale[idx]
    right_scale_val = scale[idx + 1]
    vv = (v - left_scale_val) / (right_scale_val - left_scale_val)

    # get the triplet of three values in [0,1] that represent the rgb color
    # corresponding to val
    val_color01 = colors_01[idx] + vv * (colors_01[idx + 1] - colors_01[idx])
    val_color_0255 = list(map(np.uint8, 255 * val_color01))

    return "rgb" + str(tuple(val_color_0255))


def _scale_color_values(color_values):
    """Scale all columns in the color_values array to be between 0 and 1.

    Parameters
    ----------
    color_values: 1d list or 2d array
        A 1d vector of one color value for each datapoint. If a 2d array,
        one row for each datapoint in the graph, and each column represents a
        color_value for a given point.
    """
    color_values = np.array(color_values)
    if color_values.ndim == 1:
        # Reshaping to 2-D array is required for sklearn 0.19
        color_values = color_values.reshape(-1, 1)

    color_values = color_values.astype(np.float64)
    # MinMax Scaling to be friendly to non-scaled input.
    scaler = preprocessing.MinMaxScaler()
    color_values = scaler.fit_transform(color_values)

    # "Scaler might have floating point issues, 1.0000...0002". Force max and min
    color_values[color_values > 1] = 1
    color_values[color_values < 0] = 0

    if color_values.shape[1] == 1:
        color_values = color_values.ravel()

    return color_values

def _format_meta(graph, color_function_name, node_color_function, custom_meta=None):
    n = [l for l in graph["nodes"].values()]
    n_unique = len(set([i for s in n for i in s]))

    if custom_meta is None:
        custom_meta = graph["meta_data"]

        if "clusterer" in custom_meta.keys():
            clusterer = custom_meta["clusterer"]
            custom_meta["clusterer"] = _to_html_format(clusterer)

        if "projection" in custom_meta.keys():
            projection = custom_meta["projection"]
            custom_meta["projection"] = _to_html_format(projection)

    mapper_summary = {
        "custom_meta": custom_meta,
        "color_function_name": color_function_name,
        "node_color_function": node_color_function,
        "n_nodes": len(graph["nodes"]),
        "n_edges": sum([len(l) for l in graph["links"].values()]),
        "n_total": sum([len(l) for l in graph["nodes"].values()]),
        "n_unique": n_unique,
    }

    return mapper_summary

@deprecated_alias(color_function='color_values')
def _format_mapper_data(
    graph,
    color_values,
    node_color_function,
    X,
    X_names,
    lens,
    lens_names,
    custom_tooltips,
    nbins=10,
    colorscale=None,
):
    """
    Parameters
    ----------
    color_values: 1d or 2d array
        Should have one column for each vector of datapoint color values

    node_color_function: string or 1d array
        a single string or a 1d array of string names of np function(s) to use to calcaulate node color
    """
    if colorscale is None:
        colorscale = colorscale_default

    if isinstance(node_color_function, str):
        node_color_function = [node_color_function]

    color_values = np.array(color_values)
    if color_values.ndim == 1:
        color_values = color_values.reshape(-1, 1)

    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
        node_id_to_num[node_id] = i

        node_color = []
        for _node_color_function_name in node_color_function:
            _node_color = _node_color_function(member_ids, color_values, _node_color_function_name)
            if np.array(_node_color).ndim == 0:
                _node_color = [_node_color]
            if isinstance(_node_color, np.ndarray):
                _node_color = _node_color.tolist()
            node_color.append(_node_color)

        t = _type_node()
        s = _size_node(member_ids)
        tt = _format_tooltip(
            member_ids,
            custom_tooltips,
            X,
            X_names,
            lens,
            lens_names,
            color_values,
            colorscale,
            node_id,
            nbins,
        )

        n = {
            "id": "",
            "name": node_id,
            "color": node_color,
            "type": t,
            "size": s,
            "tooltip": tt,
        }
        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in enumerate(graph["links"].items()):
        for linked_node_id in linked_node_ids:
            json_dict["links"].append(
                {
                    "source": node_id_to_num[node_id],
                    "target": node_id_to_num[linked_node_id],
                    "width": _size_link_width(graph, node_id, linked_node_id),
                }
            )
    return json_dict


def _build_histogram(data, colorscale=None, nbins=10):
    """ Build histogram of data based on values of color_values
    """
    if colorscale is None:
        colorscale = colorscale_default

    # TODO: we should weave this method of handling colors into the normal _build_histogram and combine both functions
    colorscale = _colors_to_rgb(colorscale)

    h_min, h_max = 0, 1
    hist, bin_edges = np.histogram(data, range=(h_min, h_max), bins=nbins)
    bin_mids = np.mean(np.array(list(zip(bin_edges, bin_edges[1:]))), axis=1)

    histogram = []
    max_bucket_value = max(hist)
    sum_bucket_value = sum(hist)
    for bar, mid in zip(hist, bin_mids):
        height = np.floor(((bar / max_bucket_value) * 100) + 0.5)
        perc = round((bar / sum_bucket_value) * 100.0, 1)
        color = _map_val2color(mid, 0.0, 1.0, colorscale)

        histogram.append({"height": height, "perc": perc, "color": color})

    return histogram


@deprecated_alias(color_function='color_values')
def _graph_data_distribution(graph, color_values, node_color_function, colorscale, nbins=10):

    node_averages = []
    for node_id, member_ids in graph["nodes"].items():
        node_color = _node_color_function(member_ids, color_values, node_color_function)
        node_averages.append(node_color)

    node_averages = np.array(node_averages)
    if node_averages.ndim > 1:
        histogram = []
        for node_averages_column in node_averages.T:
            _histogram = _build_histogram(node_averages_column, colorscale=colorscale, nbins=nbins)
            histogram.append(_histogram)
    else:
        histogram = _build_histogram(node_averages, colorscale=colorscale, nbins=nbins)
    return histogram


def _format_cluster_statistics(member_ids, X, X_names):
    # TODO: Cache X_mean and X_std for all clusters.
    # TODO: replace long tuples with named tuples.
    # TODO: Name all the single letter variables.
    # TODO: remove duplication between above_stats and below_stats
    # TODO: Should we only show variables that are much above or below the mean?

    cluster_data = {"above": [], "below": [], "size": len(member_ids)}

    cluster_stats = ""
    if X is not None:
        # List vs. numpy handling: cast to numpy array
        if isinstance(X_names, list):
            X_names = np.array(X_names)
        # Defaults when providing no X_names
        if X_names.shape[0] == 0:
            X_names = np.array(["f_%s" % (i) for i in range(X.shape[1])])

        # be explicit about the allowed sparse formats
        if scipy.sparse.issparse(X):
            if X.format not in ["csr", "csc"]:
                raise ValueError(
                    "sparse matrix format must be csr or csc but found {}".format(X.format))

        # wrap cluster_X_mean, X_mean, and X_std in np.array(---).squeeze()
        # to get the same treatment for dense and sparse arrays
        cluster_X_mean = np.array(
            np.mean(X[member_ids], axis=0)
        ).squeeze()
        X_mean = np.array(
            np.mean(X, axis=0)
        ).squeeze()
        X_std = np.array(
            # use StandardScaler as a way to get std for dense or sparse array
            np.sqrt(preprocessing.StandardScaler(with_mean=False).fit(X).var_)
        ).squeeze()

        above_mean = cluster_X_mean > X_mean
        std_m = np.sqrt((cluster_X_mean - X_mean) ** 2) / X_std

        stat_zip = list(
            zip(
                std_m,
                X_names,
                X_mean,
                cluster_X_mean,
                above_mean,
                X_std,
            )
        )
        stats = sorted(stat_zip, reverse=True)

        above_stats = [a for a in stats if bool(a[4]) is True]
        below_stats = [a for a in stats if bool(a[4]) is False]

        if len(above_stats) > 0:
            for s, f, i, c, a, v in above_stats[:5]:
                cluster_data["above"].append(
                    {"feature": f, "mean": round(c, 3), "std": round(s, 1)}
                )

        if len(below_stats) > 0:
            for s, f, i, c, a, v in below_stats[:5]:
                cluster_data["below"].append(
                    {"feature": f, "mean": round(c, 3), "std": round(s, 1)}
                )

    return cluster_data


def _format_projection_statistics(member_ids, lens, lens_names):
    projection_data = []

    if lens is not None:
        if isinstance(lens_names, list):
            lens_names = np.array(lens_names)

        # Create defaults when providing no lens_names
        if lens_names.shape[0] == 0:
            lens_names = np.array(["p_%s" % (i) for i in range(lens.shape[1])])

        means_v = np.mean(lens[member_ids], axis=0)
        maxs_v = np.max(lens[member_ids], axis=0)
        mins_v = np.min(lens[member_ids], axis=0)

        for name, mean_v, max_v, min_v in zip(lens_names, means_v, maxs_v, mins_v):
            projection_data.append(
                {
                    "name": name,
                    "mean": round(mean_v, 3),
                    "max": round(max_v, 3),
                    "min": round(min_v, 3),
                }
            )

    return projection_data


def _tooltip_components(
    member_ids,
    X,
    X_names,
    lens,
    lens_names,
    color_values,
    node_ID,
    colorscale,
    nbins=10,
):
    projection_stats = _format_projection_statistics(member_ids, lens, lens_names)
    cluster_stats = _format_cluster_statistics(member_ids, X, X_names)

    member_histogram = []
    for color_values_vector in color_values.T:
        _member_histogram = _build_histogram(
            color_values_vector[member_ids], colorscale=colorscale, nbins=nbins
        )
        member_histogram.append(_member_histogram)

    return projection_stats, cluster_stats, member_histogram


def _format_tooltip(
    member_ids,
    custom_tooltips,
    X,
    X_names,
    lens,
    lens_names,
    color_values,
    colorscale,
    node_ID,
    nbins,
):
    # TODO: Allow customization in the form of aggregate per node and per entry in node.
    # TODO: Allow users to turn off tooltip completely.

    custom_tooltips = (
        custom_tooltips[member_ids] if custom_tooltips is not None else member_ids
    )

    # list will render better than numpy arrays
    custom_tooltips = list(custom_tooltips)

    projection_stats, cluster_stats, histogram = _tooltip_components(
        member_ids,
        X,
        X_names,
        lens,
        lens_names,
        color_values,
        node_ID,
        colorscale,
        nbins,
    )

    tooltip_data = {
        'projection_stats': projection_stats,
        'cluster_stats': cluster_stats,
        'custom_tooltips': custom_tooltips,
        'histogram': histogram,
        'dist_label': "Member",
        'node_id': node_ID,
    }

    return tooltip_data

def _render_d3_vis(
    title,
    mapper_summary,
    histogram,
    mapper_data,
    colorscale
    ):
    # Find the module absolute path and locate templates
    module_root = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(module_root), undefined=StrictUndefined)

    # Find the absolute module path and the static files
    js_path = os.path.join(os.path.dirname(__file__), "static", "kmapper.js")
    with open(js_path, "r") as f:
        js_text = f.read()

    css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
    with open(css_path, "r") as f:
        css_text = f.read()

    if np.array(histogram).ndim == 1:
        histogram = [histogram]

    # Jinja default json serializer can't handle np arrays; provide custom encoding
    def my_dumper(obj, **kwargs):
        def np_encoder(object, **kwargs):
            if isinstance(object, np.generic):
                return np.asscalar(object)
        return json.dumps(obj, default=np_encoder, **kwargs)

    env.policies['json.dumps_function'] = my_dumper

    # Render the Jinja template, filling fields as appropriate
    html = env.get_template("base.html").render(
        title=title,
        mapper_summary=mapper_summary,
        histogram=histogram,
        dist_label="Node",
        mapper_data=mapper_data,
        colorscale=colorscale,
        js_text=js_text,
        css_text=css_text,
    )

    return html

def _node_color_function(member_ids, color_values, function_name='mean'):
    return getattr(np, function_name)(color_values[member_ids], axis=0)


def _size_node(member_ids):
    return int(np.log(len(member_ids) + 1) + 1)


def _type_node():
    return "circle"


def _size_link_width(graph, node_id, linked_node_id):
    return 1
