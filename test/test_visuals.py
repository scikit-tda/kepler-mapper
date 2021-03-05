import os
import numbers
import json
import pytest

import numpy as np
import scipy.sparse
from sklearn.datasets import make_circles
from kmapper import KeplerMapper

from kmapper import visuals
from kmapper.visuals import (
    _scale_color_values,
    _format_meta,
    _format_mapper_data,
    _map_val2color,
    _build_histogram,
    _graph_data_distribution,
    _node_color_function,
    _to_html_format,
)
from kmapper.utils import _test_raised_deprecation_warning
import warnings


from jinja2 import Environment, FileSystemLoader


np.random.seed(1)


"""
    Interested in rebuilding the API of kepler mapper so it is more intuitive


    Should Kepler Mapper be split into two objects?
    I don't get how distance_matrix works


    The visualize method should have sane defaults.

    Tooltips
        - [x] Tooltips should default to showing the ID of data point in each node.
        - Tooltips should be able to be disabled.
        - [was done already?] Tooltips should be able to show aggregate data for each node.
        - [copy and pastable] Tooltips should easily be able to export the data.

    Graph
        - Graph should be able to be frozen.
        - Graph should be able to switch between multiple coloring functions.
        - Should be able to remove nodes from graph (so you can clean out noise)
        - Edits should be able to be saved. Can re-export the html file so you can open it in the same state.
        - Color funcs should be easier to use.
        - Should be able to choose any D3 palette
        - [x] Cold is low, hot is high.

    Style:
        - [x] 'inverse_X' should just be called X
        - [x] More of the html stuff should be in the jinja2 stuff.
        - If running from source, should be able to run offline

    Map
        - Move all of these arguments into the init method

"""


@pytest.fixture
def jinja_env():
    # Find the module absolute path and locate templates
    module_root = os.path.join(os.path.dirname(__file__), "../kmapper/templates")
    env = Environment(loader=FileSystemLoader(module_root))
    return env


@pytest.fixture
def default_colorscale():
    colorscale = [
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
    return colorscale

class TestVisualHelpers:
    def test_color_values_type(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        graph = {"nodes": nodes}

        n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
        color_values = np.arange(n_samples)
        color_values = _scale_color_values(color_values)

        assert type(color_values) == np.ndarray
        assert min(color_values) == 0
        assert max(color_values) == 1

    def test_color_values_scaled(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        graph = {"nodes": nodes}

        cv = np.array([6, 5, 4, 3, 2, 1])
        color_values = _scale_color_values(cv)

        # np.testing.assert_almost_equal(min(color_values), 0)
        # np.testing.assert_almost_equal(
        #     max(color_values), 1
        # ), "Scaler might have floating point issues, 1.0000...0002"

        # _build_histogram in visuals.py assumes/needs this
        assert min(color_values) == 0
        assert max(color_values) == 1

    def test__scale_color_values_many_columns(self):
        cv1 = np.array([6, 5, 4, 3, 2, 1])
        cv2 = np.array([1, 2, 3, 4, 5, 6])
        cv = np.column_stack([cv1, cv2])
        color_values = _scale_color_values(cv)
        assert color_values.shape[1] == 2

    def test_node_averages_multiple_color_value_vectors(self):
        nodes = {"a": [0, 1, 2], "b": [3, 4, 5]}
        graph = {"nodes": nodes, "links": {}}

        n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
        color_values_1 = np.arange(n_samples)
        color_values_2 = np.flip(color_values_1)
        color_values = np.column_stack((color_values_1, color_values_2))
        color_values = _scale_color_values(color_values)

        # (Pdb) color_values
        # array([[0. , 1. ],
        #        [0.2, 0.8],
        #        [0.4, 0.6],
        #        [0.6, 0.4],
        #        [0.8, 0.2],
        #        [1. , 0. ]])

        X = np.arange(n_samples).reshape(-1,1)
        lens = np.copy(X)

        graph_data = _format_mapper_data(
            graph=graph,
            color_values=color_values,
            node_color_function='mean',
            X=None,
            X_names=[],
            lens=lens,
            lens_names=[],
            custom_tooltips=None,
            )

        nodes = { node['name']:node for node in graph_data['nodes'] }
        assert len(nodes['a']['color']) == 2
        assert np.array(nodes['a']['tooltip']['histogram']).shape[0] == 2

        np.testing.assert_almost_equal(np.array([.2, .8]), nodes['a']['color'])
        np.testing.assert_almost_equal(np.array([.8, .2]), nodes['b']['color'])


    def test_color_function_names_unequal_exception(self):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        color_values = lens[:, 0]

        cv1 = np.array(lens)
        cv2 = np.flip(cv1)
        cv = np.column_stack([cv1, cv2])
        with pytest.raises(Exception) as excinfo:
            mapper.visualize(graph, color_values=cv, color_function_name='hotdog')
        assert "Must be equal" in str(excinfo.value)

        with pytest.raises(Exception) as excinfo:
            color_values = mapper.visualize(graph, color_values=cv, color_function_name=['hotdog','hotdog','hotdiggitydog'])
        assert "Must be equal" in str(excinfo.value)

    def test_no_color_values_many_color_function_exception(self):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        with pytest.raises(Exception) as excinfo:
            color_values = mapper.visualize(graph, color_values=None, color_function_name=['hotdog','hotdog','hotdiggitydog'])
        assert "Refusing to proceed" in str(excinfo.value)

    def test_no_color_values_one_color_function_no_exception_yes_warning(self):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            color_values = mapper.visualize(graph, color_values=None, color_function_name=['hotdog'])
        assert "unexpected" in str(w[-1].message)

    def test_color_hist_matches_nodes(self):
        """ The histogram colors dont seem to match the node colors,
            this should confirm the colors will match and we need to look at the
            javascript instead.
        """

        color_values = np.array([0.55] * 10 + [0.0] * 10)
        member_ids = [1, 2, 3, 4, 5, 6]
        hist = _build_histogram(color_values[member_ids])
        c = round(_node_color_function(member_ids, color_values), 2)
        single_bar = [bar for bar in hist if bar["perc"] == 100.0]

        assert len(single_bar) == 1
        assert _map_val2color(c, 0.0, 1.0) == single_bar[0]["color"]

    def test_node_color_function_works(self):
        color_values = np.arange(20)
        member_ids = np.arange(color_values.shape[0])
        assert _node_color_function(member_ids, color_values, 'mean') == np.mean(color_values) == 9.5
        assert _node_color_function(member_ids, color_values, 'median') == np.median(color_values) == 9.5
        assert _node_color_function(member_ids, color_values, 'max') == np.max(color_values) == 19
        assert _node_color_function(member_ids, color_values, 'min') == np.min(color_values) == 0

    def test_color_values_size(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6, 7, 8, 9]}
        graph = {"nodes": nodes}

        n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
        color_values = np.arange(n_samples)
        color_values = _scale_color_values(color_values)

        assert len(color_values) == len(nodes["a"]) + len(nodes["b"]) + 1

    def test_format_meta(self):
        mapper = KeplerMapper()
        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        color_function_name = ['Row number']

        fmt = _format_meta(graph, color_function_name)
        assert fmt["n_nodes"] == len(graph["nodes"])

        assert "n_edges" in fmt.keys()
        assert "n_total" in fmt.keys()

        del fmt["custom_meta"]
        vals = fmt.values()
        for v in vals:
            assert isinstance(v, numbers.Number)

    def test_format_meta_with_meta(self):
        mapper = KeplerMapper()
        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        color_function_name = ['Row number']
        node_color_function = 'mean'

        cm = "My custom_meta"
        fmt = _format_meta(graph, color_function_name, node_color_function, cm)
        assert fmt["custom_meta"] == cm

    def test_format_meta(self, sc):
        mapper_summary = _format_meta(sc, "foo", "bar", "Nada custom meta")
        assert mapper_summary["custom_meta"] == "Nada custom meta"
        assert (
            mapper_summary["n_total"] <= 300 and mapper_summary["n_total"] >= 200
        ), "Some points become repeated in multiple nodes."

    def test_node_color_function_must_be_np_function(self, sc):
        mapper = KeplerMapper()

        with pytest.raises(AttributeError, match=r".*must be a function available on `numpy` class.*"):
            mapper.visualize(sc, node_color_function=['yinz'])



    def test_to_html_format(self):
        res = _to_html_format("a\nb\n\n\\n\n")
        assert "\n" not in res
        assert "<br>" in res

    def test_visualize_one_color_function(self):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        color_values = lens[:, 0]

        mapper.visualize(graph, color_values=color_values, color_function_name=['hotdog'])

    def test_visualize_multiple_color_functions(self):
        """ convenience test for generating a vis with multiple color_values"""
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        color_values = lens[:, 0]

        cv1 = np.array(lens)
        cv2 = np.flip(cv1)
        cv = np.column_stack([cv1, cv2])
        mapper.visualize(graph, color_values=cv, color_function_name=['hotdog','hotdiggitydog'])

    def test_visualize_graph_with_cluster_stats_above_below(self):
        mapper = KeplerMapper()
        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 10
        lens = mapper.fit_transform(X, projection=[0])
        graph = mapper.map(lens, X)
        output = mapper.visualize(
            graph,
            X=X,
            lens=X
        )
        # then, visually inspect mapper_visualization_output.html

    def test_color_function_deprecated_replaced(self, default_colorscale):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        color_values = lens[:, 0]
        node_color_function = 'mean'
        inverse_X = data
        projected_X = lens
        projected_X_names = ["projected_%s" % (i) for i in range(projected_X.shape[1])]
        inverse_X_names = ["inverse_%s" % (i) for i in range(inverse_X.shape[1])]
        custom_tooltips = np.array(["customized_%s" % (l) for l in labels])

        # https://docs.python.org/3/library/warnings.html#testing-warnings
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # kmapper.visualize
            html = mapper.visualize(graph, color_function=lens, color_function_name='lens[:, 0]')
            _test_raised_deprecation_warning(w)

            # visuals._format_mapper_data
            graph_data = _format_mapper_data(
                graph=graph,
                color_function=color_values,
                node_color_function=node_color_function,
                X=inverse_X,
                X_names=inverse_X_names,
                lens=projected_X,
                lens_names=projected_X_names,
                custom_tooltips=custom_tooltips,
            )
            _test_raised_deprecation_warning(w)

            # visuals._graph_data_distribution
            histogram = _graph_data_distribution(graph, color_function=lens, node_color_function=node_color_function, colorscale=default_colorscale)
            _test_raised_deprecation_warning(w)




    def test__format_mapper_data(self):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        color_values = lens[:, 0]
        node_color_function = 'mean'
        inverse_X = data
        projected_X = lens
        projected_X_names = ["projected_%s" % (i) for i in range(projected_X.shape[1])]
        inverse_X_names = ["inverse_%s" % (i) for i in range(inverse_X.shape[1])]
        custom_tooltips = np.array(["customized_%s" % (l) for l in labels])

        graph_data = _format_mapper_data(
            graph,
            color_values,
            node_color_function,
            inverse_X,
            inverse_X_names,
            projected_X,
            projected_X_names,
            custom_tooltips,
        )
        # print(graph_data)
        # Dump to json so we can easily tell what's in it.
        graph_data = json.dumps(graph_data)

        # TODO test more properties!
        assert "name" in graph_data
        assert """cube2_cluster0""" in graph_data
        assert """projected_0""" in graph_data
        assert """inverse_0""" in graph_data

        assert """customized_""" in graph_data

    def test_histogram(self):
        data = np.random.random((100, 1))
        hist = visuals._build_histogram(data)
        assert isinstance(hist, list)
        assert isinstance(hist[0], dict)
        assert len(hist) == 10

    def test_cluster_stats(self):
        X = np.random.random((1000, 3))
        ids = np.random.choice(20, 1000)

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert isinstance(cluster_data, dict)
        assert cluster_data["size"] == len(ids)

    def test_cluster_stats_sparse_csr(self):
        X = scipy.sparse.random(1000, 3, density=1.0, format="csr")
        ids = np.random.choice(20, 1000)

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert isinstance(cluster_data, dict)
        assert cluster_data["size"] == len(ids)

    def test_cluster_stats_sparse_csc(self):
        X = scipy.sparse.random(1000, 3, density=1.0, format="csc")
        ids = np.random.choice(20, 1000)

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert isinstance(cluster_data, dict)
        assert cluster_data["size"] == len(ids)

    def test_cluster_stats_sparse_coo(self):
        X = scipy.sparse.random(1000, 3, density=1.0, format="coo")
        ids = np.random.choice(20, 1000)

        with pytest.raises(ValueError, match=r".*sparse matrix format must be csr or csc.*"):
            cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

    def test_cluster_stats_above(self):
        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 10

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert len(cluster_data["above"]) >= 1
        assert cluster_data["above"][0]["feature"] == "a"
        assert cluster_data["above"][0]["mean"] == 10

    def test_cluster_stats_below(self):
        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 0

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert len(cluster_data["below"]) >= 1
        assert cluster_data["below"][0]["feature"] == "a"
        assert cluster_data["below"][0]["mean"] == 0

    def test_cluster_stats_with_no_names(self):
        # This would be the default.

        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 0

        cluster_data = visuals._format_cluster_statistics(ids, X, [])

        assert len(cluster_data["below"]) >= 1
        assert cluster_data["below"][0]["feature"] == "f_0"
        assert cluster_data["below"][0]["mean"] == 0


class TestVisualizeIntegration:
    def test_empty_graph_warning(self):
        mapper = KeplerMapper()

        graph = {"nodes": {}}
        with pytest.raises(Exception):
            mapper.visualize(graph)

    def test_visualize_standalone_same(self, tmpdir):
        """ ensure that the visualization is not dependent on the actual mapper object.
        """
        mapper = KeplerMapper()

        file = tmpdir.join("output.html")

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz1 = mapper.visualize(graph, path_html=file.strpath)

        new_mapper = KeplerMapper()
        viz2 = new_mapper.visualize(graph, path_html=file.strpath)

        assert viz1 == viz2

    def test_file_written(self, tmpdir):
        mapper = KeplerMapper()

        file = tmpdir.join("output.html")

        data = np.random.rand(1000, 2)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz = mapper.visualize(graph, path_html=file.strpath)

        assert file.read() == viz
        assert len(tmpdir.listdir()) == 1, "file was written to"

    def test_file_not_written(self, tmpdir):
        mapper = KeplerMapper(verbose=1)

        file = tmpdir.join("output.html")

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz = mapper.visualize(graph, path_html=file.strpath, save_file=False)

        assert len(tmpdir.listdir()) == 0, "file was never written to"
        # assert file.read() != viz


class TestColorhandling:
    def test_map_val2color_on_point(self, default_colorscale):
        """ This function takes a val, a min and max, and a color scale, and finds the color the val should be """

        for v, color in default_colorscale:
            c = _map_val2color(v, 0.0, 1.0, default_colorscale)
            assert c == color

    def test_mid_val2color(self, default_colorscale):
        expected = int((72 + 68) / 2), int((35 + 1) / 2), int((116 + 85) / 2)
        expected_str = (
            "rgb("
            + str(expected[0])
            + ", "
            + str(expected[1])
            + ", "
            + str(expected[2])
            + ")"
        )
        c = _map_val2color(0.05, 0.0, 1.0, default_colorscale)
        assert c == expected_str
