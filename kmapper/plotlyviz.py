from __future__ import division
from .utils import deprecated_alias

import numpy as np

from .visuals import (
    _scale_color_values,
    _size_node,
    _format_cluster_statistics,
    _node_color_function,
    _format_meta,
    _graph_data_distribution,
    _tooltip_components,
)

try:
    import igraph as ig
    import plotly.graph_objs as go
    import ipywidgets as ipw
    import plotly.io as pio
except ImportError:
    print(
        """To use the plotly visualization tools, you must have the packages igraph, plotly, and ipywidgets installed in your environment."""
        """ It looks like at least one of these is missing.  Please install again with"""
        """\n\n\t`pip install igraph plotly ipywidgets`\n\nand try again"""
    )
    raise


default_colorscale = [
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


def mpl_to_plotly(cmap, n_entries):
    h = 1.0 / (n_entries - 1)
    pl_colorscale = []
    for k in range(n_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append(
            [round(k * h, 2), "rgb" + str((C[0], C[1], C[2]))]
        )  # Python 2.7+
        # pl_colorscale.append([round(k*h, 2), f'rgb({C[0]}, {C[1]}, {C[2]})']) # Python 3.6+
    return pl_colorscale


@deprecated_alias(color_function="color_values")
def plotlyviz(
    scomplex,
    colorscale=None,
    title="Kepler Mapper",
    graph_layout="kk",
    color_values=None,
    color_function_name=None,
    node_color_function="mean",
    dashboard=False,
    graph_data=False,
    factor_size=3,
    edge_linewidth=1.5,
    node_linecolor="rgb(200,200,200)",
    width=600,
    height=500,
    bgcolor="rgba(240, 240, 240, 0.95)",
    left=10,
    bottom=35,
    summary_height=300,
    summary_width=600,
    summary_left=20,
    summary_right=20,
    hist_left=25,
    hist_right=25,
    member_textbox_width=800,
    filename=None,
):
    """
    Visualizations and dashboards for kmapper graphs using Plotly. This method is suitable for use in Jupyter notebooks.


    The generated FigureWidget can be updated (by performing a restyle or relayout). For example, let us add a title
    to the colorbar (the name of the color function, if any),
    and set the title font size. To perform these updates faster, Plotly 3.+ provides a context manager that batches up all data and layout updates:

    To display more info on the generated kmapper-graph, define two more FigureWidget(s):
    the global node distribution figure, and a dummy figure
    that displays info on the  algorithms involved in getting the graph from data, as well as  sklearn  class instances.

    A FigureWidget has event listeners for hovering, clicking or selecting. Using the first one for `fw_graph`
    we   define, via the function `hovering_widgets()`, widgets that display the node distribution, when the node is hovered over, and two textboxes for the cluster size and the member ids/labels of the hovered node members.



    Parameters
    -----------

    scomplex: dict
        Simplicial complex is the output from the KeplerMapper `map` method.

    title: str
        Title of output graphic

    graph_layout: igraph layout;
        recommended 'kk' (kamada-kawai) or 'fr' (fruchterman-reingold)

    colorscale:
         Plotly colorscale(colormap) to color graph nodes

    dashboard: bool, default is False
        If true, display complete dashboard of node information

    graph_data: bool, default is False
        If true, display graph metadata

    factor_size: double, default is 3
        a factor for the node size

    edge_linewidth : double, default is 1.5
    node_linecolor: color str, default is "rgb(200,200,200)"
    width: int, default is 600,
    height: int, default is 500,
    bgcolor: color str, default is "rgba(240, 240, 240, 0.95)",
    left: int, default is 10,
    bottom: int, default is 35,
    summary_height: int, default is 300,
    summary_width: int, default is 600,
    summary_left: int, default is 20,
    summary_right: int, default is 20,
    hist_left: int, default is 25,
    hist_right: int, default is 25,
    member_textbox_width: int, default is 800,
    filename: str, default is None
        if filename is given, the graphic will be saved to that file.


    Returns
    ---------
    result: plotly.FigureWidget
        A FigureWidget that can be shown or editted. See the Plotly Demo notebook for examples of use.

    """

    if not colorscale:
        colorscale = default_colorscale

    kmgraph, mapper_summary, n_color_distribution = get_mapper_graph(
        scomplex,
        colorscale=colorscale,
        color_values=color_values,
        color_function_name=color_function_name,
        node_color_function=node_color_function,
    )

    annotation = get_kmgraph_meta(mapper_summary)

    plgraph_data = plotly_graph(
        kmgraph,
        graph_layout=graph_layout,
        colorscale=colorscale,
        factor_size=factor_size,
        edge_linewidth=edge_linewidth,
        node_linecolor=node_linecolor,
    )

    layout = plot_layout(
        title=title,
        width=width,
        height=height,
        annotation_text=annotation,
        bgcolor=bgcolor,
        left=left,
        bottom=bottom,
    )
    result = go.FigureWidget(data=plgraph_data, layout=layout)

    if color_function_name:
        with result.batch_update():
            result.data[1].marker.colorbar.title = color_function_name
            result.data[1].marker.colorbar.titlefont.size = 10

    if dashboard or graph_data:
        fw_hist = node_hist_fig(n_color_distribution, left=hist_left, right=hist_right)
        fw_summary = summary_fig(
            mapper_summary,
            width=summary_width,
            height=summary_height,
            left=summary_left,
            right=summary_right,
        )

        fw_graph = result
        result = hovering_widgets(
            kmgraph, fw_graph, member_textbox_width=member_textbox_width
        )

        if graph_data:
            result = ipw.VBox([fw_graph, ipw.HBox([fw_summary, fw_hist])])

    if filename:
        pio.write_image(result, filename)

    return result


@deprecated_alias(color_function="color_values")
def scomplex_to_graph(
    simplicial_complex,
    color_values,
    X,
    X_names,
    lens,
    lens_names,
    custom_tooltips,
    colorscale,
    node_color_function="mean",
):
    color_values = np.array(color_values)

    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(simplicial_complex["nodes"].items()):
        node_id_to_num[node_id] = i
        projection_stats, cluster_stats, member_histogram = _tooltip_components(
            member_ids, X, X_names, lens, lens_names, color_values, i, colorscale
        )
        node_color = _node_color_function(member_ids, color_values, node_color_function)
        if isinstance(node_color, np.ndarray):
            node_color = node_color.tolist()
        n = {
            "id": i,
            "name": node_id,
            "member_ids": member_ids,
            "color": node_color,
            "size": _size_node(member_ids),
            "cluster": cluster_stats,
            "distribution": member_histogram,
            "projection": projection_stats,
            "custom_tooltips": custom_tooltips,
        }

        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in enumerate(simplicial_complex["links"].items()):
        for linked_node_id in linked_node_ids:
            lnk = {
                "source": node_id_to_num[node_id],
                "target": node_id_to_num[linked_node_id],
            }

            json_dict["links"].append(lnk)

    return json_dict


@deprecated_alias(color_function="color_values")
def get_mapper_graph(
    simplicial_complex,
    color_values=None,
    color_function_name=None,
    node_color_function="mean",
    colorscale=None,
    custom_tooltips=None,
    custom_meta=None,
    X=None,
    X_names=None,
    lens=None,
    lens_names=None,
):
    """Generate data for mapper graph visualization and annotation.

    Parameters
    ----------
    simplicial_complex : dict
        Simplicial complex is the output from the KeplerMapper `map` method.

    Returns
    -------
    the graph dictionary in a json representation, the mapper summary
    and the node_distribution

    Example
    -------

    >>> kmgraph,  mapper_summary, n_distribution = get_mapper_graph(simplicial_complex)

    """

    if not colorscale:
        colorscale = default_colorscale

    if not len(simplicial_complex["nodes"]) > 0:
        raise Exception(
            "A mapper graph should have more than 0 nodes. This might be because your clustering algorithm might be too sensitive and be classifying all points as noise."
        )

    if color_values is None:
        # If no color_values provided we color by row order in data set
        n_samples = (
            np.max([i for s in simplicial_complex["nodes"].values() for i in s]) + 1
        )
        color_values = np.arange(n_samples)
        color_function_name = ["Row number"]

    color_values = _scale_color_values(color_values)

    if X_names is None:
        X_names = []

    if lens_names is None:
        lens_names = []

    json_graph = scomplex_to_graph(
        simplicial_complex,
        color_values,
        X,
        X_names,
        lens,
        lens_names,
        custom_tooltips,
        colorscale=colorscale,
        node_color_function=node_color_function,
    )

    colorf_distribution = _graph_data_distribution(
        simplicial_complex, color_values, node_color_function, colorscale
    )

    mapper_summary = _format_meta(
        simplicial_complex,
        color_function_name=color_function_name,
        node_color_function=node_color_function,
        custom_meta=custom_meta,
    )

    return json_graph, mapper_summary, colorf_distribution


def plotly_graph(
    kmgraph,
    graph_layout="kk",
    colorscale=None,
    showscale=True,
    factor_size=3,
    edge_linecolor="rgb(180,180,180)",
    edge_linewidth=1.5,
    node_linecolor="rgb(255,255,255)",
    node_linewidth=1.0,
):
    """Generate Plotly data structures that represent the mapper graph

    Parameters
    ----------
    kmgraph: dict representing the mapper graph,
             returned by the function get_mapper_graph()
    graph_layout: igraph layout; recommended 'kk' (kamada-kawai)
                  or 'fr' (fruchterman-reingold)
    colorscale: a Plotly colorscale(colormap) to color graph nodes
    showscale: boolean to display or not the colorbar
    factor_size: a factor for the node size

    Returns
    -------
    The plotly traces (dicts) representing the graph edges and nodes
    """

    if not colorscale:
        colorscale = default_colorscale

    # define an igraph.Graph instance of n_nodes
    n_nodes = len(kmgraph["nodes"])
    if n_nodes == 0:
        raise ValueError("Your graph has 0 nodes")
    G = ig.Graph(n=n_nodes)
    links = [(e["source"], e["target"]) for e in kmgraph["links"]]
    G.add_edges(links)
    layt = G.layout(graph_layout)

    hover_text = [node["name"] for node in kmgraph["nodes"]]
    color_vals = [node["color"] for node in kmgraph["nodes"]]
    node_size = np.array(
        [factor_size * node["size"] for node in kmgraph["nodes"]], dtype=int
    )
    Xn, Yn, Xe, Ye = _get_plotly_data(links, layt)

    edge_trace = dict(
        type="scatter",
        x=Xe,
        y=Ye,
        mode="lines",
        line=dict(color=edge_linecolor, width=edge_linewidth),
        hoverinfo="none",
    )

    node_trace = dict(
        type="scatter",
        x=Xn,
        y=Yn,
        mode="markers",
        marker=dict(
            size=node_size.tolist(),
            color=color_vals,
            opacity=1.0,
            colorscale=colorscale,
            showscale=showscale,
            line=dict(color=node_linecolor, width=node_linewidth),
            colorbar=dict(thickness=20, ticklen=4, x=1.01, tickfont=dict(size=10)),
        ),
        text=hover_text,
        hoverinfo="text",
    )

    return [edge_trace, node_trace]


def get_kmgraph_meta(mapper_summary):
    """Extract info from mapper summary to be displayed below the graph plot"""
    d = mapper_summary["custom_meta"]
    meta = (
        "<b>N_cubes:</b> "
        + str(d["n_cubes"])
        + " <b>Perc_overlap:</b> "
        + str(d["perc_overlap"])
    )
    meta += (
        "<br><b>Nodes:</b> "
        + str(mapper_summary["n_nodes"])
        + " <b>Edges:</b> "
        + str(mapper_summary["n_edges"])
        + " <b>Total samples:</b> "
        + str(mapper_summary["n_total"])
        + " <b>Unique_samples:</b> "
        + str(mapper_summary["n_unique"])
    )

    return meta


def plot_layout(
    title="TDA KMapper",
    width=600,
    height=600,
    bgcolor="rgba(255, 255, 255, 1)",
    annotation_text=None,
    annotation_x=0,
    annotation_y=-0.01,
    top=100,
    left=60,
    right=60,
    bottom=60,
):
    """Set the plotly layout

    Parameters
    ----------
    width, height: integers
        setting  width and height of plot window
    bgcolor: string,
        rgba or hex color code for the background color
    annotation_text: string
        meta data to be displayed
    annotation_x & annotation_y:
        The coordinates of the point where we insert the annotation; the negative sign for y coord points output that annotation is inserted below the plot
    """
    pl_layout = dict(
        title=title,
        font=dict(size=12),
        showlegend=False,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
        plot_bgcolor=bgcolor,
        margin=dict(t=top, b=bottom, l=left, r=right),
    )

    if annotation_text is None:
        return pl_layout
    else:
        annotations = [
            dict(
                showarrow=False,
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=annotation_x,
                y=annotation_y,
                align="left",
                xanchor="left",
                yanchor="top",
                font=dict(size=12),
            )
        ]
        pl_layout.update(annotations=annotations)
        return pl_layout


def node_hist_fig(
    node_color_distribution,
    title="Graph Node Distribution",
    width=400,
    height=300,
    top=60,
    left=25,
    bottom=60,
    right=25,
    bgcolor="rgb(240,240,240)",
    y_gridcolor="white",
):
    """Define the plotly plot representing the node histogram

    Parameters
    ----------
    node_color_distribution: list of dicts describing the _build_histogram
    width, height: integers -  width and height of the histogram FigureWidget
    left, top, right, bottom: ints; number of pixels around the FigureWidget
    bgcolor: rgb of hex color code for the figure background color
    y_gridcolor: rgb of hex color code for the yaxis y_gridcolor

    Returns
    -------
    FigureWidget object representing the histogram of the graph nodes
    """

    text = [
        "{perc}%".format(**locals())
        for perc in [d["perc"] for d in node_color_distribution]
    ]

    pl_hist = go.Bar(
        y=[d["height"] for d in node_color_distribution],
        marker=dict(color=[d["color"] for d in node_color_distribution]),
        text=text,
        hoverinfo="y+text",
    )

    hist_layout = dict(
        title=title,
        width=width,
        height=height,
        font=dict(size=12),
        xaxis=dict(showline=True, zeroline=False, showgrid=False, showticklabels=False),
        yaxis=dict(showline=False, gridcolor=y_gridcolor, tickfont=dict(size=10)),
        bargap=0.01,
        margin=dict(l=left, r=right, b=bottom, t=top),
        hovermode="x",
        plot_bgcolor=bgcolor,
    )

    return go.FigureWidget(data=[pl_hist], layout=hist_layout)


def summary_fig(
    mapper_summary,
    width=600,
    height=500,
    top=60,
    left=20,
    bottom=60,
    right=20,
    bgcolor="rgb(240,240,240)",
):
    """Define a dummy figure that displays info on the algorithms and
    sklearn class instances or methods used

    Returns a FigureWidget object representing the figure
    """
    text = _text_mapper_summary(mapper_summary)

    data = [
        dict(
            type="scatter",
            x=[0, width],
            y=[height, 0],
            mode="text",
            text=[text, ""],
            textposition="bottom right",
            hoverinfo="none",
        )
    ]

    layout = dict(
        title="Algorithms and scikit-learn objects/methods",
        width=width,
        height=height,
        font=dict(size=12),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, height + 5]),
        margin=dict(t=top, b=bottom, l=left, r=right),
        plot_bgcolor=bgcolor,
    )

    return go.FigureWidget(data=data, layout=layout)


def hovering_widgets(
    kmgraph,
    graph_fw,
    ctooltips=False,
    width=400,
    height=300,
    top=100,
    left=50,
    bgcolor="rgb(240,240,240)",
    y_gridcolor="white",
    member_textbox_width=200,
):
    """Defines the widgets that display the distribution of each node on hover
        and the members of each nodes

    Parameters
    ----------
    kmgraph: the kepler-mapper graph dict returned by `get_mapper_graph()``
    graph_fw: the FigureWidget representing the graph
    ctooltips: boolean; if True/False the node["custom_tooltips"]/"member_ids"
    are passed to member_textbox

    width, height, top refer to the figure
    size and position of the hovered node distribution

    Returns
    -------
    a box containing the graph figure, the figure of the hovered node
    distribution, and the textboxes displaying the cluster size and  member_ids
    or custom tooltips for hovered node members
    """
    fnode = kmgraph["nodes"][0]
    fwc = node_hist_fig(
        fnode["distribution"],
        title="Cluster Member Distribution",
        width=width,
        height=height,
        top=top,
        left=left,
        bgcolor=bgcolor,
        y_gridcolor=y_gridcolor,
    )
    clust_textbox = ipw.Text(
        value="{:d}".format(fnode["cluster"]["size"]),
        description="Cluster size:",
        disabled=False,
        continuous_update=True,
    )

    clust_textbox.layout = dict(margin="10px 10px 10px 10px", width="200px")

    member_textbox = ipw.Textarea(
        value=", ".join(str(x) for x in fnode["member_ids"])
        if not ctooltips
        else ", ".join(str(x) for x in fnode["custom_tooltips"]),
        description="Members:",
        disabled=False,
        continuous_update=True,
    )

    member_textbox.layout = dict(
        margin="5px 5px 5px 10px", width=str(member_textbox_width) + "px"
    )

    def do_on_hover(trace, points, state):
        if not points.point_inds:
            return
        ind = points.point_inds[0]  # get the index of the hovered node
        node = kmgraph["nodes"][ind]
        # on hover do:
        with fwc.batch_update():  # update data in the cluster member histogr
            fwc.data[0].text = [
                "{:.1f}%".format(d["perc"]) for d in node["distribution"]
            ]
            fwc.data[0].y = [d["height"] for d in node["distribution"]]
            fwc.data[0].marker.color = [d["color"] for d in node["distribution"]]

        clust_textbox.value = "{:d}".format(node["cluster"]["size"])
        member_textbox.value = (
            ", ".join(str(x) for x in node["member_ids"])
            if not ctooltips
            else ", ".join(str(x) for x in node["custom_tooltips"])
        )

    trace = graph_fw.data[1]
    trace.on_hover(do_on_hover)
    return ipw.VBox([ipw.HBox([graph_fw, fwc]), clust_textbox, member_textbox])


def _get_plotly_data(E, coords):
    # E : the list of tuples representing the graph edges
    # coords: list of node coordinates assigned by igraph.Layout
    N = len(coords)
    Xnodes = [coords[k][0] for k in range(N)]  # x-coordinates of nodes
    Ynodes = [coords[k][1] for k in range(N)]  # y-coordnates of nodes

    Xedges = []
    Yedges = []
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def _text_mapper_summary(mapper_summary):
    d = mapper_summary["custom_meta"]
    text = "<br><b>Projection: </b>" + d["projection"]
    text += (
        "<br><b>Clusterer: </b>" + d["clusterer"] + "<br><b>Scaler: </b>" + d["scaler"]
    )
    if "color_function" in d.keys():
        text += "<br><b>Color function: </b>" + d["color_function"]

    return text


def _hover_format(member_ids, custom_tooltips, X, X_names, lens, lens_names):
    cluster_data = _format_cluster_statistics(member_ids, X, X_names)
    tooltip = ""
    custom_tooltips = (
        custom_tooltips[member_ids] if custom_tooltips is not None else member_ids
    )
    val_size = cluster_data["size"]
    tooltip += "{val_size}".format(**locals())
    return tooltip
