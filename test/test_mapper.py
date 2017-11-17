import pytest
import numpy as np

from kmapper import KeplerMapper

from kmapper.kmapper import Cover


class TestVisualize():
    def test_visualize_standalone(self):
        # visualize will run on a fresh mapper object
        mapper = KeplerMapper()

        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        new_mapper = KeplerMapper()
        viz = new_mapper.visualize(graph)


class TestLinker():
    def test_finds_a_link(self):
        mapper = KeplerMapper()

        groups = {"a": [1,2,3,4], "b":[1,2,3,4]}
        links = mapper._create_links(groups)

        assert "a" in links or "b" in links
        assert links["a"] == ["b"] or links["b"] == ["a"]

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
    def test_cube_count(self):
        data = np.arange(30).reshape(10,3)
        c = Cover(data, nr_cubes=10)
        cubes = c.cubes

        assert len(cubes) == 10**3

    def test_cube_dim(self):

        data = np.arange(30).reshape(10,3)
        c = Cover(data, nr_cubes=10)
        cubes = c.cubes

        assert all( len(cube) == 3 for cube in cubes)

    def test_single_dim(self):
        data = np.arange(10).reshape(10,1)
        c = Cover(data, nr_cubes=10)
        cubes = c.cubes

        assert all( len(cube) == 1 for cube in cubes)

    def test_chunk_dist(self):
        data = np.arange(100).reshape(10,10)

        cover = Cover(data, nr_cubes=10)
        chunks = list(cover.chunk_dist)

        assert all(i == 9 for i in chunks)

    def test_nr_dimensions(self):
        data = np.arange(30).reshape(10,3)

        c = Cover(data, nr_cubes=10)

        assert c.nr_dimensions == 3

    def test_bound_is_min(self):
        data = np.arange(100).reshape(10,10)

        c = Cover(data, nr_cubes=10)

        bounds = zip(c.d, range(10))
        assert all(b[0] == b[1] for b in bounds)

    def test_entries_even(self):
        data = np.arange(20).reshape(20,1)

        cover = Cover(data, nr_cubes=10)
        cubes = cover._cube_coordinates_all()

        for cube in cubes:
            entries = cover.find_entries(data, cube)

            assert len(entries) >= 2

    def test_entries_in_correct_cubes(self):
        data = np.arange(20).reshape(20,1)

        cover = Cover(data, nr_cubes=10)
        cubes = cover._cube_coordinates_all()

        entries = [cover.find_entries(data, cube) for cube in cubes]

        # inside of each cube is there. Sometimes the edges don't line up.
        for i in range(10):
            assert data[2*i] in entries[i]
            assert data[2*i+1] in entries[i]

    def test_cubes_overlap(self):
        data = np.arange(20).reshape(20,1)

        cover = Cover(data, nr_cubes=10)
        cubes = cover._cube_coordinates_all()

        entries = []
        for cube in cubes:
            # turn singleton lists into individual elements
            res = [i[0] for i in cover.find_entries(data, cube)]
            entries.append(res)

        for i,j in zip(range(9), range(1,10)):
            assert set(entries[i]).union(set(entries[j]))


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
        assert graph["meta_graph"] == "custom"

    def test_projection(self):
        atol = 0.1 # accomodate scaling, values are in (0,1), but will be scaled slightly

        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.fit_transform(data, projection=[0,1])
        np.testing.assert_allclose(lens, data[:,:2], atol=atol)

        lens = mapper.fit_transform(data, projection=[0])
        np.testing.assert_allclose(lens, data[:,:1], atol=atol)
