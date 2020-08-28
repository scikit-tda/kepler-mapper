import pytest
import numpy as np
import kmapper as km
from kmapper import draw_matplotlib



@pytest.fixture
def mapper():
    mapper = km.KeplerMapper(verbose=0)
    data = np.random.rand(100, 2)
    graph = mapper.map(data)
    return graph


class TestDrawMPL:
    def test_mapper_input(self, mapper):
        draw_matplotlib(mapper)

    def test_nx_input(self, mapper):
        draw_matplotlib(km.to_networkx(mapper))
