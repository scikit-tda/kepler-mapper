import numpy as np

import networkx as nx

from kmapper import to_networkx, to_nx, KeplerMapper


class TestNetworkx:
    def test_convertions(self):
        mapper = KeplerMapper(verbose=0)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

        g = to_networkx(graph)
        assert isinstance(g, nx.Graph)

    def test_membership(self):
        mapper = KeplerMapper(verbose=0)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)
        n, m = list(graph["nodes"].items())[0]
        g = to_networkx(graph)
        assert g.nodes[n]["membership"] == m

    def test_nx_alias(self):
        mapper = KeplerMapper(verbose=0)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

        g = to_nx(graph)
        assert isinstance(g, nx.Graph)
