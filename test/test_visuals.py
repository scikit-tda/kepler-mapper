import numpy as np

from kmapper import KeplerMapper

from kmapper.visuals import init_color_function


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

        color_function = init_color_function(graph)

        assert min(color_function) == 0
        assert max(color_function) == 1

    def test_color_function_size(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6, 7, 8, 9]}
        graph = {"nodes": nodes}

        color_function = init_color_function(graph)

        assert len(color_function) == len(nodes['a']) + len(nodes['b']) + 1


class TestVisualizeIntegration():
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

        data = np.random.rand(1000, 10)
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
