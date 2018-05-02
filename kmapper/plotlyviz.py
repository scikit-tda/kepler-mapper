import igraph as ig
import numpy as np


def get_plotly_data(E, coords):
    # E is the list of tuples representing the graph edges
    # coords is the list of node coordinates assigned by igraph.Layout
    N = len(coords)
    Xnodes = [coords[k][0] for k in range(N)]  # x-coordinates of nodes
    Ynodes = [coords[k][1] for k in range(N)]  # y-coordnates of nodes

    Xedges = []
    Yedges = []
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def plotly_graph(kmgraph, graph_layout='kk', colorscale='Viridis',
                 reversescale=False, showscale=True, factor_size=2,
                 keep_kmtooltips=True,
                 edge_linecolor='rgb(200,200,200)', edge_linewidth=1.5):
    # kmgraph: a dict returned by the method visualize, when path_html=None
    # graph_layout: an igraph layout; recommended 'kk' or 'fr'
    # factor_size: a factor for the node size
    # keep_tooltip: True  to keep the tooltips assigned by kmapper;
    # False, when kmapper tooltips contains projection statistic

    # define an igraph.Graph instance of n_nodes
    n_nodes = len(kmgraph['nodes'])
    if n_nodes == 0:
        raise ValueError('Your graph has 0 nodes')
    G = ig.Graph(n=n_nodes)
    links = [(e['source'], e['target']) for e in kmgraph['links']]
    G.add_edges(links)
    layt = G.layout(graph_layout)

    if keep_kmtooltips:
        tooltips = [node['name'] + '<br>' + node['tooltip']
                    for node in kmgraph['nodes']]
    else:
        tooltips = [node['name'] for node in kmgraph['nodes']]

    color_vals = [node['color'] for node in kmgraph['nodes']]
    size = np.array([factor_size * node['size'] for node in kmgraph['nodes']],
                    dtype=np.int)
    Xn, Yn, Xe, Ye = get_plotly_data(links, layt)
    edges_trace = dict(type='scatter',
                       x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color=edge_linecolor,
                                 width=edge_linewidth),
                       hoverinfo='none')

    nodes_trace = dict(type='scatter',
                       x=Xn,
                       y=Yn,
                       mode='markers',
                       marker=dict(symbol='dot',
                                   size=size,
                                   color=color_vals,
                                   colorscale=colorscale,
                                   showscale=showscale,
                                   reversescale=reversescale,
                                   colorbar=dict(thickness=20,
                                                 ticklen=4)),
                       text=tooltips,
                       hoverinfo='text')

    return [edges_trace, nodes_trace]


def plot_layout(title='TDA KMapper', width=700, height=700,
                bgcolor='rgba(20,20,20, 0.8)', annotation_text=False,
                annotation_x=0, annotation_y=-0.07):
    # width, height: plot window width, height
    # bgcolor: plot background color
    # annotation_text: meta data to be displayed
    # annotation_x, annotation_y are the coordinates of the
    # point where we insert the annotation

    # set the axes style; No axes!
    axis = dict(showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    pl_layout = dict(title=title,
                     font=dict(size=12),
                     showlegend=False,
                     autosize=False,
                     width=width,
                     height=height,
                     xaxis=dict(axis),
                     yaxis=dict(axis),
                     hovermode='closest',
                     plot_bgcolor=bgcolor)

    if annotation_text is None:
        return pl_layout
    else:
        annotations = [dict(showarrow=False,
                            text=annotation_text,
                            xref='paper',
                            yref='paper',
                            x=annotation_x,
                            y=annotation_y,
                            xanchor='left',
                            yanchor='top',
                            font=dict(size=14))]
        pl_layout.update(annotations=annotations)
        return pl_layout
