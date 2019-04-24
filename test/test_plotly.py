""" 

Tests to ensure the plotly data preparation functions work as intended.

"""
import pytest

import json
import numpy as np

from kmapper import KeplerMapper
from kmapper.plotlyviz import (
    build_histogram,
    default_colorscale,
    get_mapper_graph,
    _map_val2color,
    plotly_graph,
    format_meta,
    _to_html_format,
)


@pytest.fixture
def sc():
    X = np.random.random((200, 5))
    km = KeplerMapper()
    lens = km.fit_transform(X)
    sc = km.map(lens, X)
    return sc


def test_histogram_default_colorscale():
    hist = build_histogram(np.random.random((100, 1)), default_colorscale)
    assert isinstance(hist, list)
    assert isinstance(hist[0], dict)
    assert len(hist) == len(default_colorscale) - 1


def test_kepler_to_graph(sc):

    json_graph, mapper_summary, colorf_distribution = get_mapper_graph(sc)

    assert json.loads(json.dumps(json_graph)) == json_graph
    assert isinstance(mapper_summary, dict)
    assert isinstance(colorf_distribution, list)

def test_kepler_to_graph_with_colorscale(sc):

    json_graph, mapper_summary, colorf_distribution = get_mapper_graph(sc, colorscale=default_colorscale)

    assert json.loads(json.dumps(json_graph)) == json_graph
    assert isinstance(mapper_summary, dict)
    assert isinstance(colorf_distribution, list)


def test_plotly_graph(sc):
    edge_trace, node_trace = plotly_graph(get_mapper_graph(sc)[0])
    assert isinstance(edge_trace, dict)
    assert isinstance(node_trace, dict)


def test_format_meta(sc):
    mapper_summary = format_meta(sc, "Nada custom meta", "foo")
    assert mapper_summary["custom_meta"] == "Nada custom meta"
    assert (
        mapper_summary["n_total"] <= 300 and mapper_summary["n_total"] >= 200
    ), "Some points become repeated in multiple nodes."


def test_to_html_format():
    res = _to_html_format("a\nb\n\n\\n\n")
    assert "\n" not in res
    assert "<br>" in res
