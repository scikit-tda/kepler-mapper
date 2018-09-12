from __future__ import division
from .visuals import (init_color_function,
                      _size_node,
                      _format_projection_statistics,
                      _format_cluster_statistics)

import igraph as ig
import numpy as np
import plotly.graph_objs as go
from ast import literal_eval
import ipywidgets as ipw


colorscale = [[0.0, 'rgb(68, 1, 84)'],  # Viridis
              [0.1, 'rgb(72, 35, 116)'],
              [0.2, 'rgb(64, 67, 135)'],
              [0.3, 'rgb(52, 94, 141)'],
              [0.4, 'rgb(41, 120, 142)'],
              [0.5, 'rgb(32, 144, 140)'],
              [0.6, 'rgb(34, 167, 132)'],
              [0.7, 'rgb(68, 190, 112)'],
              [0.8, 'rgb(121, 209, 81)'],
              [0.9, 'rgb(189, 222, 38)'],
              [1.0, 'rgb(253, 231, 36)']]


def pl_build_histogram(data, colorscale):
    # Build histogram of data based on values of color_function
    if colorscale[0][1][0] == '#':
        plotly_colors = np.array(colorscale)[:, 1].tolist()
        for k, hexcode in enumerate(plotly_colors):
            hexcode = hexcode.lstrip('#')
            hex_len = len(hexcode)
            step = hex_len // 3
            colorscale[k][1] = 'rgb' + str(tuple(int(hexcode[j:j + step], 16)
                                           for j in range(0, hex_len, step)))

    h_min, h_max = 0, 1
    hist, bin_edges = np.histogram(data, range=(h_min, h_max), bins=10)
    bin_mids = np.mean(np.array(list(zip(bin_edges, bin_edges[1:]))), axis=1)

    histogram = []
    max_bucket_value = max(hist)
    sum_bucket_value = sum(hist)
    for bar, mid in zip(hist, bin_mids):
        height = np.floor(((bar / max_bucket_value) * 100) + 0.5)
        perc = round((bar / sum_bucket_value) * 100., 1)
        color = _map_val2color(mid, 0., 1., colorscale)

        histogram.append({
            'height': height,
            'perc': perc,
            'color': color
        })

    return histogram


def pl_graph_data_distribution(graph, color_function, colorscale):

    node_averages = []
    for node_id, member_ids in graph["nodes"].items():
        member_colors = color_function[member_ids]
        node_averages.append(np.mean(member_colors))

    histogram = pl_build_histogram(node_averages, colorscale)

    return histogram


def scomplex_to_graph(simplicial_complex, color_function, X, X_names,
                      lens, lens_names, custom_tooltips, colorscale):

    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in \
            enumerate(simplicial_complex["nodes"].items()):
        node_id_to_num[node_id] = i
        projection_stats, cluster_stats, member_histogram =\
            _pl_format_tooltip(member_ids,
                               custom_tooltips,
                               X,
                               X_names,
                               lens,
                               lens_names,
                               color_function,
                               i,
                               colorscale)
        n = {"id": i,
             "name": node_id,
             "member_ids": member_ids,
             "color": _pl_color_function(member_ids, color_function),
             "size": _size_node(member_ids),
             "cluster": cluster_stats,
             "distribution": member_histogram,
             "projection": projection_stats,
             "custom_tooltips": custom_tooltips}

        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in\
            enumerate(simplicial_complex["links"].items()):
        for linked_node_id in linked_node_ids:
            lnk = {"source": node_id_to_num[node_id],
                   "target": node_id_to_num[linked_node_id]}

            json_dict["links"].append(lnk)

    return json_dict


def pl_format_meta(graph, color_function_name, custom_meta=None):
    n = [l for l in graph["nodes"].values()]
    n_unique = len(set([i for s in n for i in s]))

    if custom_meta is None:
        custom_meta = graph['meta_data']
        clusterer = custom_meta['clusterer']
        custom_meta['clusterer'] = clusterer.replace('\n', '<br>')
        if 'projection' in custom_meta.keys():
            projection = custom_meta['projection']
            custom_meta['projection'] = projection.replace('\n', '<br>')
        if color_function_name is not None:
            custom_meta['color_function'] = color_function_name
    mapper_summary = {
        "custom_meta": custom_meta,
        "n_nodes": len(graph["nodes"]),
        "n_edges": sum([len(l) for l in graph["links"].values()]),
        "n_total": sum([len(l) for l in graph["nodes"].values()]),
        "n_unique": n_unique
    }

    return mapper_summary


def get_mapper_graph(simplicial_complex,
                     color_function=None,
                     color_function_name=None,
                     colorscale=colorscale,
                     custom_tooltips=None,
                     custom_meta=None,
                     X=None,
                     X_names=[],
                     lens=None,
                     lens_names=[]):
    """Generate data for mapper graph visualization and annotation.

    Parameters
    ----------
    simplicial_complex : dict
        Simplicial complex is the output from the KeplerMapper `map` method.
    Returns
    -------
    the graph dictionary in  a json representation, the mapper summary
    and the node_distribution

    Example
    -------

    >>> kmgraph,  mapper_summary, n_distribution = \
                                         get_mapper_graph(simplicial_complex)

    """
    if not len(simplicial_complex['nodes']) > 0:
        raise Exception("A mapper graph should have more than 0 nodes")

    color_function = init_color_function(simplicial_complex, color_function)

    json_graph = scomplex_to_graph(simplicial_complex, color_function, X,
                                   X_names, lens, lens_names, custom_tooltips,
                                   colorscale=colorscale)
    colorf_distribution = pl_graph_data_distribution(
                                                    simplicial_complex,
                                                    color_function, colorscale)
    mapper_summary = pl_format_meta(simplicial_complex,
                                    color_function_name, custom_meta)

    return json_graph,  mapper_summary, colorf_distribution


def plotly_graph(kmgraph, graph_layout='kk', colorscale=colorscale,
                 showscale=True, factor_size=3,
                 edge_linecolor='rgb(200,200,200)', edge_linewidth=1.5,
                 node_linecolor='rgb(240,240,240)', node_linewidth=0.5):
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
    # define an igraph.Graph instance of n_nodes
    n_nodes = len(kmgraph['nodes'])
    if n_nodes == 0:
        raise ValueError('Your graph has 0 nodes')
    G = ig.Graph(n=n_nodes)
    links = [(e['source'], e['target']) for e in kmgraph['links']]
    G.add_edges(links)
    layt = G.layout(graph_layout)

    hover_text = [node['name'] for node in kmgraph['nodes']]
    color_vals = [node['color'] for node in kmgraph['nodes']]
    node_size = np.array([factor_size * node['size'] for node in
                          kmgraph['nodes']], dtype=np.int)
    Xn, Yn, Xe, Ye = _get_plotly_data(links, layt)

    edge_trace = dict(type='scatter',
                      x=Xe,
                      y=Ye,
                      mode='lines',
                      line=dict(color=edge_linecolor,
                                width=edge_linewidth),
                      hoverinfo='none')

    node_trace = dict(type='scatter',
                      x=Xn,
                      y=Yn,
                      mode='markers',
                      marker=dict(size=node_size.tolist(),
                                  color=color_vals,
                                  colorscale=colorscale,
                                  showscale=showscale,
                                  line=dict(color=node_linecolor,
                                            width=node_linewidth),
                                  colorbar=dict(thickness=20,
                                                ticklen=4,
                                                x=1.01,
                                                tickfont=dict(size=10))),
                      text=hover_text,
                      hoverinfo='text')

    return [edge_trace, node_trace]


def get_kmgraph_meta(mapper_summary):
    # Extract info from mapper summary to be displayed below the graph plot
    d = mapper_summary['custom_meta']
    meta = "<b>N_cubes:</b> " + str(d['n_cubes']) +\
           " <b>Perc_overlap:</b> " + str(d['perc_overlap'])
    meta += "<br><b>Nodes:</b> " + str(mapper_summary['n_nodes']) +\
            " <b>Edges:</b> " + str(mapper_summary['n_edges']) +\
            " <b>Total samples:</b> " + str(mapper_summary['n_total']) +\
            " <b>Unique_samples:</b> " + str(mapper_summary['n_unique'])

    return meta


def plot_layout(title='TDA KMapper', width=600, height=600,
                bgcolor='rgba(10,10,10, 0.95)', annotation_text=None,
                annotation_x=0, annotation_y=-0.01, top=100, left=60,
                right=60, bottom=60):
    """Set the plotly layout
    Parameters
    ----------
    width, height: integers setting  width and height of plot window
    bgcolor: rgb, rgba or hex color code for the background color
    annotation_text: string; meta data to be displayed
    annotation_x, annotation_y are the coordinates of the
    point where we insert the annotation; the negative sign for y coord
    points output that  annotation is inserted below the plot
    """
    pl_layout = dict(title=title,
                     font=dict(size=12),
                     showlegend=False,
                     autosize=False,
                     width=width,
                     height=height,
                     xaxis=dict(visible=False),
                     yaxis=dict(visible=False),
                     hovermode='closest',
                     plot_bgcolor=bgcolor,
                     margin=dict(t=top, b=bottom, l=left, r=right)
                     )

    if annotation_text is None:
        return pl_layout
    else:
        annotations = [dict(showarrow=False,
                            text=annotation_text,
                            xref='paper',
                            yref='paper',
                            x=annotation_x,
                            y=annotation_y,
                            align='left',
                            xanchor='left',
                            yanchor='top',
                            font=dict(size=12))]
        pl_layout.update(annotations=annotations)
        return pl_layout


def node_hist_fig(node_color_distribution, title='Graph Node Distribution',
                  width=400, height=300, top=60, left=25, bottom=60, right=25,
                  bgcolor='rgb(240,240,240)', y_gridcolor='white'):
    """Define the plotly plot representing the node histogram
    Parameters
    ----------
    node_color_distribution: list of dicts describing the build_histogram
    width, height: integers -  width and height of the histogram FigureWidget
    left, top, right, bottom: ints; number of pixels around the FigureWidget
    bgcolor: rgb of hex color code for the figure background color
    y_gridcolor: rgb of hex color code for the yaxis y_gridcolor

    Returns
    -------
    FigureWidget object representing the histogram of the graph nodes
    """

    text = ["{perc}%".format(**locals()) for perc in
            [d['perc'] for d in node_color_distribution]]

    pl_hist = go.Bar(y=[d['height'] for d in node_color_distribution],
                     marker=dict(color=[d['color'] for d in
                                        node_color_distribution]),
                     text=text,
                     hoverinfo='y+text')

    hist_layout = dict(title=title,
                       width=width, height=height,
                       font=dict(size=12),
                       xaxis=dict(showline=True, zeroline=False,
                                  showgrid=False, showticklabels=False),
                       yaxis=dict(showline=False, gridcolor=y_gridcolor,
                                  tickfont=dict(size=10)),
                       bargap=0.01,
                       margin=dict(l=left, r=right, b=bottom, t=top),
                       hovermode='x',
                       plot_bgcolor=bgcolor)

    return go.FigureWidget(data=[pl_hist], layout=hist_layout)


def summary_fig(mapper_summary, width=600, height=500, top=60,
                left=20, bottom=60, right=20, bgcolor='rgb(240,240,240)'):
    """Define a dummy figure that displays info on the algorithms and
       sklearn class instances or methods used

       Returns a FigureWidget object representing the figure
    """
    text = _text_mapper_summary(mapper_summary)

    data = [dict(type='scatter',
                 x=[0,   width],
                 y=[height,   0],
                 mode='text',
                 text=[text, ''],
                 textposition='bottom right',
                 hoverinfo='none')]

    layout = dict(title='Algorithms and scikit-learn objects/methods',
                  width=width, height=height,
                  font=dict(size=12),
                  xaxis=dict(visible=False),
                  yaxis=dict(visible=False, range=[0, height+5]),
                  margin=dict(t=top, b=bottom, l=left, r=right),
                  plot_bgcolor=bgcolor)

    return go.FigureWidget(data=data, layout=layout)


def hovering_widgets(kmgraph, graph_fw, ctooltips=False, width=400,
                     height=300, top=100, left=50,
                     bgcolor='rgb(240,240,240)',
                     y_gridcolor='white', member_textbox_width=200):
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
    fnode = kmgraph['nodes'][0]
    fwc = node_hist_fig(fnode['distribution'],
                        title='Cluster Member Distribution',
                        width=width, height=height, top=top, left=left,
                        bgcolor=bgcolor, y_gridcolor=y_gridcolor)
    clust_textbox = ipw.Text(value='{:d}'.format(fnode['cluster']['size']),
                             description='Cluster size:',
                             disabled=False,
                             continuous_update=True)

    clust_textbox.layout = dict(margin='10px 10px 10px 10px', width='200px')

    member_textbox = ipw.Textarea(value=', '.join(str(x) for x in
                                  fnode['member_ids']) if not ctooltips
                                  else ', '.join(str(x)
                                  for x in fnode['custom_tooltips']),
                                  description='Members:',
                                  disabled=False,
                                  continuous_update=True)

    member_textbox.layout = dict(margin='5px 5px 5px 10px',
                                 width=str(member_textbox_width)+'px')

    def do_on_hover(trace, points, state):
        if not points.point_inds:
            return
        ind = points.point_inds[0]  # get the index of the hovered node
        node = kmgraph['nodes'][ind]
        # on hover do:
        with fwc.batch_update():  # update data in the cluster member histogr
            fwc.data[0].text = ['{:.1f}%'.format(d['perc']) for d in
                                node['distribution']]
            fwc.data[0].y = [d['height'] for d in node['distribution']]
            fwc.data[0].marker.color = [d['color'] for d in
                                        node['distribution']]

        clust_textbox.value = '{:d}'.format(node['cluster']['size'])
        member_textbox.value = ', '.join(str(x) for x in node['member_ids'])\
                               if not ctooltips else\
                               ', '.join(str(x) for x in
                                         node['custom_tooltips'])
    trace = graph_fw.data[1]
    trace.on_hover(do_on_hover)
    return ipw.VBox([ipw.HBox([graph_fw, fwc]), clust_textbox, member_textbox])


def _map_val2color(val, vmin, vmax, colorscale):
    # maps a value val in [vmin, vmax] to the corresponding color in
    # the colorscale
    # returns the rgb color code of that color

    if vmin >= vmax:
        raise ValueError('vmin should be < vmax')

    plotly_scale = list(map(float, np.array(colorscale)[:, 0]))
    plotly_colors = np.array(colorscale)[:, 1]

    colors_01 = np.array(list(map(literal_eval, [color[3:] for color in
                                                 plotly_colors])))/255.

    v = (val - vmin) / float((vmax - vmin))  # val is mapped to v in[0,1]

    idx = 0
    # sequential search for the two   consecutive indices idx, idx+1 such that
    # v belongs to the interval  [plotly_scale[idx], plotly_scale[idx+1]
    while(v > plotly_scale[idx+1]):
        idx += 1
    left_scale_val = plotly_scale[idx]
    right_scale_val = plotly_scale[idx + 1]
    vv = (v - left_scale_val) / (right_scale_val - left_scale_val)

    # get the triplet of three values in [0,1] that represent the rgb color
    # corresponding to val
    val_color01 = colors_01[idx] + vv * (colors_01[idx + 1] - colors_01[idx])
    val_color_0255 = list(map(np.uint8, 255*val_color01))

    return 'rgb'+str(tuple(val_color_0255))


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

    d = mapper_summary['custom_meta']
    text = "<br><b>Projection: </b>" + d['projection']
    text += "<br><b>Clusterer: </b>" + d['clusterer'] +\
            "<br><b>Scaler: </b>" + d['scaler']
    if 'color_function' in d.keys():
        text += "<br><b>Color function: </b>" + d['color_function']

    return text


def _pl_format_tooltip(member_ids, custom_tooltips, X,
                       X_names, lens, lens_names,
                       color_function, node_ID, colorscale):

    custom_tooltips = custom_tooltips[member_ids] if custom_tooltips\
                                                is not None else member_ids

    custom_tooltips = list(custom_tooltips)

    projection_stats = _format_projection_statistics(member_ids, lens,
                                                     lens_names)
    cluster_stats = _format_cluster_statistics(member_ids, X, X_names)
    member_histogram = pl_build_histogram(color_function[member_ids],
                                          colorscale)

    return projection_stats, cluster_stats, member_histogram


def _hover_format(member_ids, custom_tooltips, X,
                  X_names, lens, lens_names):
    cluster_data = _format_cluster_statistics(member_ids, X, X_names)
    tooltip = ''
    custom_tooltips = custom_tooltips[member_ids] if custom_tooltips\
                          is not None else member_ids
    val_size = cluster_data['size']
    tooltip += "{val_size}".format(**locals())
    return tooltip


def _pl_color_function(member_ids, color_function):
    return np.mean(color_function[member_ids])
