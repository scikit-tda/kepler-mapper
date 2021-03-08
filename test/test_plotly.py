"""

Tests to ensure the plotly data preparation functions work as intended.

"""
import pytest

import json
import numpy as np
from sklearn.datasets import make_circles

from kmapper import KeplerMapper
from kmapper.plotlyviz import (
    _build_histogram,
    default_colorscale,
    get_mapper_graph,
    _map_val2color,
    plotly_graph,
    scomplex_to_graph,
)
from kmapper.utils import _test_raised_deprecation_warning

import warnings


def test_histogram_default_colorscale():
    hist = _build_histogram(np.random.random((100, 1)), default_colorscale)
    assert isinstance(hist, list)
    assert isinstance(hist[0], dict)
    assert len(hist) == len(default_colorscale) - 1


def test_kepler_to_graph(sc):

    json_graph, mapper_summary, colorf_distribution = get_mapper_graph(sc)

    assert json.loads(json.dumps(json_graph)) == json_graph
    assert isinstance(mapper_summary, dict)
    assert isinstance(colorf_distribution, list)


def test_kepler_to_graph_with_colorscale(sc):

    json_graph, mapper_summary, colorf_distribution = get_mapper_graph(
        sc, colorscale=default_colorscale
    )

    assert json.loads(json.dumps(json_graph)) == json_graph
    assert isinstance(mapper_summary, dict)
    assert isinstance(colorf_distribution, list)


def test_plotly_graph(sc):
    edge_trace, node_trace = plotly_graph(get_mapper_graph(sc)[0])
    assert isinstance(edge_trace, dict)
    assert isinstance(node_trace, dict)


def test_color_function_deprecated_replaced():
    km = KeplerMapper()
    X, labels = make_circles(1000, random_state=0)
    lens = km.fit_transform(X, projection=[0])
    color_values = lens[:, 0]
    node_color_function = "mean"
    sc = km.map(lens, X)
    X_names = []
    lens_names = []
    custom_tooltips = np.array(["customized_%s" % (l) for l in labels])

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # TODO: plotlyviz.plotlyviz

        # plotlyviz.get_mapper_graph
        json_graph, mapper_summary, colorf_distribution = get_mapper_graph(
            sc, color_function=color_values, node_color_function=node_color_function
        )
        _test_raised_deprecation_warning(w)

        # plotlyviz.scomplex_to_graph
        _ = scomplex_to_graph(
            simplicial_complex=sc,
            color_function=color_values,
            node_color_function=node_color_function,
            X=X,
            X_names=X_names,
            lens=lens,
            lens_names=lens_names,
            custom_tooltips=custom_tooltips,
            colorscale=default_colorscale,
        )
        _test_raised_deprecation_warning(w)
