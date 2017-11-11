import pytest
import numpy as np

from kmapper import KeplerMapper
#from km import KeplerMapper


class TestLinker():
    def test_finds_a_link(self):
        mapper = KeplerMapper()

        groups = {"a": [1,2,3,4], "b":[1,2,3,4]}
        links = mapper._create_links(groups)

        assert "a" in links
        assert links["a"] == ["b"]

    def test_no_link(self):
        mapper = KeplerMapper()

        groups = {"a": [1,2,3,4], "b":[5,6,7]}
        links = mapper._create_links(groups)

        assert not links

    def test_pass_through_result(self):
        mapper = KeplerMapper()

        groups = {"a": [1], "b":[2]}

        res = dict()
        links = mapper._create_links(groups, res)

        assert res == links


def test_lens_size():
    mapper = KeplerMapper()

    data = np.random.rand(100, 10)
    lens = mapper.fit_transform(data)

    assert lens.shape[0] == data.shape[0]


def test_num_cubes():
    mapper = KeplerMapper()

    nr_cubes = 10
    nr_dimensions = 2
    bins = mapper._cube_coordinates_all(nr_cubes, nr_dimensions)

    assert len(bins) == nr_cubes**nr_dimensions


def test_cube_dimensions():
    mapper = KeplerMapper()

    nr_cubes = 10
    nr_dimensions = 2
    bins = mapper._cube_coordinates_all(nr_cubes, nr_dimensions)

    assert len(bins[0]) == nr_dimensions
