""" Adapt Mapper format into other common formats.

    - networkx

"""


def to_networkx(graph):
    """ Convert a Mapper 1-complex to a networkx graph.

    Parameters
    -----------

    graph: dictionary, graph object returned from `kmapper.map`

    Returns
    --------

    g: graph as networkx.Graph() object

    """

    # import here so networkx is not always required.
    import networkx as nx

    nodes = graph["nodes"].keys()
    edges = [[start, end] for start, ends in graph["links"].items() for end in ends]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    nx.set_node_attributes(g, dict(graph["nodes"]), "membership")

    g.add_edges_from(edges)

    return g


to_nx = to_networkx
