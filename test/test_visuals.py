import os
import numbers
import json
import pytest

import numpy as np
from sklearn.datasets import make_circles
from kmapper import KeplerMapper

from kmapper import visuals
from kmapper.visuals import init_color_function, format_meta, format_mapper_data
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
    module_root = os.path.join(os.path.dirname(__file__), '../kmapper/templates')
    env = Environment(loader=FileSystemLoader(module_root))
    return env


class TestVisualHelpers():
    def test_color_function_type(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        graph = {"nodes": nodes}

        color_function = init_color_function(graph)

        assert type(color_function) == np.ndarray
        assert min(color_function) == 0
        assert max(color_function) == 1

    def test_color_function_scaled(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        graph = {"nodes": nodes}

        cf = np.array([6, 5, 4, 3, 2, 1])
        color_function = init_color_function(graph, cf)

        np.testing.assert_almost_equal(min(color_function), 0)
        np.testing.assert_almost_equal(max(color_function), 1), "Scaler might have floating point issues, 1.0000...0002"

    def test_color_function_size(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6, 7, 8, 9]}
        graph = {"nodes": nodes}

        color_function = init_color_function(graph)

        assert len(color_function) == len(nodes['a']) + len(nodes['b']) + 1

    def test_format_meta(self):
        mapper = KeplerMapper()
        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        fmt = format_meta(graph)
        assert fmt['n_nodes'] == len(graph["nodes"])

        assert 'n_edges' in fmt.keys()
        assert 'n_total' in fmt.keys()

        del fmt['custom_meta']
        vals = fmt.values()
        for v in vals:
            assert isinstance(v, numbers.Number)

    def test_format_meta_with_meta(self):
        mapper = KeplerMapper()
        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        cm = "My custom_meta"
        fmt = format_meta(graph, cm)
        assert fmt['custom_meta'] == cm

    def test_format_mapper_data(self, jinja_env):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        color_function = lens[:, 0]
        inverse_X = data
        projected_X = lens
        projected_X_names = ["projected_%s" % (i) for i in range(projected_X.shape[1])]
        inverse_X_names = ["inverse_%s" % (i) for i in range(inverse_X.shape[1])]
        custom_tooltips = np.array(["customized_%s" % (l) for l in labels])

        graph_data = format_mapper_data(graph, color_function, inverse_X,
                                        inverse_X_names, projected_X, projected_X_names, custom_tooltips, jinja_env)
        # print(graph_data)
        # Dump to json so we can easily tell what's in it.
        graph_data = json.dumps(graph_data)

        # TODO test more properties!
        assert 'name' in graph_data
        assert """cube2_cluster0""" in graph_data
        assert """projected_0""" in graph_data
        assert """inverse_0""" in graph_data

        assert """customized_""" in graph_data

    def test_histogram(self):
        data = np.random.random((100, 1))
        hist = visuals.build_histogram(data)
        assert isinstance(hist, list)
        assert isinstance(hist[0], dict)
        assert len(hist) == 10

    def test_cluster_stats(self):
        X = np.random.random((1000, 3))
        ids = np.random.choice(20, 1000)

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert isinstance(cluster_data, dict)
        assert cluster_data['size'] == len(ids)

    def test_cluster_stats_above(self):
        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 10

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert len(cluster_data['above']) >= 1
        assert cluster_data['above'][0]['feature'] == 'a'
        assert cluster_data['above'][0]['mean'] == 10

    def test_cluster_stats_below(self):
        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 0

        cluster_data = visuals._format_cluster_statistics(ids, X, ["a", "b", "c"])

        assert len(cluster_data['below']) >= 1
        assert cluster_data['below'][0]['feature'] == 'a'
        assert cluster_data['below'][0]['mean'] == 0

    def test_cluster_stats_with_no_names(self):
        # This would be the default.

        X = np.ones((1000, 3))
        ids = np.random.choice(20, 1000)
        X[ids, 0] = 0

        cluster_data = visuals._format_cluster_statistics(ids, X, [])

        assert len(cluster_data['below']) >= 1
        assert cluster_data['below'][0]['feature'] == 'f_0'
        assert cluster_data['below'][0]['mean'] == 0


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

        file = tmpdir.join('output.html')

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz1 = mapper.visualize(graph, path_html=file.strpath)

        new_mapper = KeplerMapper()
        viz2 = new_mapper.visualize(graph, path_html=file.strpath)

        assert viz1 == viz2

    def test_file_written(self, tmpdir):
        mapper = KeplerMapper()

        file = tmpdir.join('output.html')

        data = np.random.rand(1000, 2)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz = mapper.visualize(graph, path_html=file.strpath)

        assert file.read() == viz
        assert len(tmpdir.listdir()) == 1, "file was written to"

    def test_file_not_written(self, tmpdir):
        mapper = KeplerMapper()

        file = tmpdir.join('output.html')

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz = mapper.visualize(graph, path_html=file.strpath, save_file=False)

        assert len(tmpdir.listdir()) == 0, "file was never written to"
        # assert file.read() != viz

