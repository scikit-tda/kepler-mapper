import pytest
import numpy as np

from kmapper import KeplerMapper

from kmapper.kmapper import Cover


class TestCuber():
    def test_cube_count(self):
        mapper = KeplerMapper()
        cubes = mapper._cube_coordinates_all(4, 3)
        assert len(cubes) == 4**3

    def test_cube_dim(self):
        mapper = KeplerMapper()
        cubes = mapper._cube_coordinates_all(4, 3)
        assert all( len(cube) == 3 for cube in cubes)

    def test_single_dim(self):
        mapper = KeplerMapper()
        cubes = mapper._cube_coordinates_all(4, 1)
        assert all( len(cube) == 1 for cube in cubes)


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

class TestCover():
    def test_chunk_dist(self):
        data = np.arange(100).reshape(10,10)

        cover = Cover(data, 10)
        chunks = list(cover.chunk_dist)
        for c in chunks:
            assert c == 9
        #assert all(i == 9 for i in chunks)

    def test_bound_is_min(self):
        data = np.arange(100).reshape(10,10)

        c = Cover(data, 10)

        bounds = zip(c.d, range(10))
        assert all(b[0] == b[1] for b in bounds)

    def test_entries(self):
        data = np.arange(100).reshape(10,2)

        cover = Cover(data, 10)
        cubes = cover._cube_coordinates_all()
        cube = cover.find_entries(data, cubes[0])



def test_lens_size():
    mapper = KeplerMapper()

    data = np.random.rand(100, 10)
    lens = mapper.fit_transform(data)

    assert lens.shape[0] == data.shape[0]

def test_map_custom_lens():
    # I think that map currently requires fit_transform to be called first
    mapper = KeplerMapper()
    data = np.random.rand(100, 2)
    graph = mapper.map(data)
    assert graph["meta_graph"] == "custom"
