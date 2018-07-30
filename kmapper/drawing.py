"""
    Methods for drawing graphs

"""

import numpy as np


def draw_matplotlib(g, ax=None, fig=None):
    """ Draw the graph as a networkx graph
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    fig = fig if fig else plt.figure()
    ax = ax if ax else plt.gca()

    if not isinstance(g, nx.Graph):
        from .adapter import to_networkx

        g = to_networkx(g)

    # Determine a fine size for nodes
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    area = width * height * fig.dpi

    n_nodes = len(g.nodes)

    # size of node should be related to area and number of nodes -- heuristic
    node_size = np.pi * area / n_nodes
    node_r = np.sqrt(node_size / np.pi)
    node_edge = node_r / 3

    pos = nx.spring_layout(g)

    nodes = nx.draw_networkx_nodes(g, node_size=node_size, pos=pos)
    edges = nx.draw_networkx_edges(g, pos=pos)
    nodes.set_edgecolor("w")
    nodes.set_linewidth(node_edge)

    plt.axis("square")
    plt.axis("off")

    return nodes
