import pytest
import numpy as np

from kmapper import KeplerMapper


class TestLogging():
    """ Simple tests that confirm map completes at each logging level
    """

    def test_runs_with_logging_0(self):
        mapper = KeplerMapper(verbose=0)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

    def test_runs_with_logging_1(self):
        mapper = KeplerMapper(verbose=1)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

    def test_runs_with_logging_2(self):
        mapper = KeplerMapper(verbose=2)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)


class TestVisualize():
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


class TestLinker():
    def test_finds_a_link(self):
        mapper = KeplerMapper()

        groups = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
        links = mapper._create_links(groups)

        assert "a" in links or "b" in links
        assert links["a"] == ["b"] or links["b"] == ["a"]

    def test_no_link(self):
        mapper = KeplerMapper()

        groups = {"a": [1, 2, 3, 4], "b": [5, 6, 7]}
        links = mapper._create_links(groups)

        assert not links

    def test_pass_through_result(self):
        mapper = KeplerMapper()

        groups = {"a": [1], "b": [2]}

        res = dict()
        links = mapper._create_links(groups, res)

        assert res == links


class TestLens():
    # TODO: most of these tests only accomodate the default option. They need to be extended to incorporate all possible transforms.
    def test_lens_size(self):
        mapper = KeplerMapper()

        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        assert lens.shape[0] == data.shape[0]

    def test_map_custom_lens(self):
        # I think that map currently requires fit_transform to be called first
        mapper = KeplerMapper()
        data = np.random.rand(100, 2)
        #import pdb; pdb.set_trace()
        graph = mapper.map(data)
        assert graph["meta_data"]["projection"] == "custom"
        assert graph["meta_data"]["scaler"] == "None"

    def test_projection(self):
        # accomodate scaling, values are in (0,1), but will be scaled slightly
        atol = 0.1

        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.fit_transform(data, projection=[0, 1])
        np.testing.assert_allclose(lens, data[:, :2], atol=atol)

        lens = mapper.fit_transform(data, projection=[0])
        np.testing.assert_allclose(lens, data[:, :1], atol=atol)
