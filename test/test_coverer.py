import pytest
import numpy as np
from sklearn import datasets

from kmapper import KeplerMapper

from kmapper.cover import Cover


@pytest.mark.parametrize("CoverClass", [Cover])
class TestCoverBasic:
    def test_cube_dim(self, CoverClass):

        data = np.arange(30).reshape(10, 3)
        c = CoverClass(n_cubes=10)
        cubes = c.define_bins(data)

        assert all(len(cube) == 2 for cube in cubes)

    def test_cube_count(self, CoverClass):
        data = np.arange(30).reshape(10, 3)
        c = CoverClass(n_cubes=10)
        cubes = c.define_bins(data)

        assert len(list(cubes)) == 10 ** 2, "idx column is ignored"

    def test_single_dim(self, CoverClass):
        data = np.arange(20).reshape(10, 2)
        c = CoverClass(n_cubes=10)
        cubes = c.define_bins(data)

        assert all(len(cube) == 1 for cube in cubes)

    def test_nr_dimensions(self, CoverClass):
        data = np.arange(30).reshape(10, 3)

        c = CoverClass(n_cubes=10)
        _ = c.define_bins(data)
        assert c.nr_dimensions == 2

    def test_entries_even(self, CoverClass):
        data = np.arange(40).reshape(20, 2)

        cover = CoverClass(n_cubes=10)
        cubes = cover.define_bins(data)

        for cube in cubes:
            entries = cover.find_entries(data, cube)
            assert len(entries) >= 2

    def test_cubes_overlap(self, CoverClass):
        data = np.arange(40).reshape(20, 2)

        cover = CoverClass(n_cubes=10)
        cubes = cover.define_bins(data)

        entries = []
        for cube in cubes:
            # turn singleton lists into individual elements
            res = [i[0] for i in cover.find_entries(data, cube)]
            entries.append(res)

        for i, j in zip(range(9), range(1, 10)):
            assert set(entries[i]).union(set(entries[j]))

    def test_complete_pipeline(self, CoverClass):
        # TODO: add a mock that asserts the cover was called appropriately.. or test number of cubes etc.
        data, _ = datasets.make_circles()

        data = data.astype(np.float64)
        mapper = KeplerMapper()
        graph = mapper.map(data, coverer=CoverClass())
        mapper.visualize(graph)


class TestCover:
    def test_diff_overlap_per_dim(self):
        data = np.random.rand(100, 10)
        c = Cover(overlap_perc=[2, 10])

    def test_define_diff_bins_per_dim(self):
        data = np.arange(30).reshape(10, 3)
        c = Cover(n_cubes=[5, 10])
        cubes = c.define_bins(data)
        assert len(list(cubes)) == 5 * 10

    def test_find_entries_runs_with_diff_bins(self):
        data = np.arange(30).reshape(10, 3)
        c = Cover(n_cubes=[5, 10])
        cubes = list(c.define_bins(data))
        _ = c.find_entries(data, cubes[0])

    def test_chunk_dist(self):
        data = np.arange(20).reshape(10, 2)

        cover = Cover(n_cubes=10)
        _ = cover.define_bins(data)
        chunks = list(cover.chunk_dist)
        # TODO: this test is really fagile and has magic number, fix.
        assert all(i == 1.8 for i in chunks)

    def test_bound_is_min(self):
        data = np.arange(30).reshape(10, 3)
        cov = Cover(n_cubes=10)
        _ = cov.define_bins(data)
        bounds = list(zip(cov.d, range(1, 10)))
        assert all(b[0] == b[1] for b in bounds)

    def test_entries_in_correct_cubes(self):
        # TODO: this test is a little hacky

        data = np.arange(40).reshape(20, 2)

        cover = Cover(n_cubes=10)
        cubes = cover.define_bins(data)
        cubes = list(cubes)
        entries = [cover.find_entries(data, cube) for cube in cubes]

        # inside of each cube is there. Sometimes the edges don't line up.
        for i in range(10):
            assert data[2 * i] in entries[i]
            assert data[2 * i + 1] in entries[i]


class TestCoverBounds:
    def test_bounds(self):
        data_vals = np.arange(40).reshape(20, 2)
        data = np.zeros((20, 3))
        data[:, 0] = np.arange(20, dtype=int)  # Index row
        data[:, 1:3] = data_vals

        limits = np.array([[np.float("inf"), np.float("inf")], [-10, 100]])
        cover = Cover(n_cubes=10, limits=limits)
        cubes = cover.define_bins(data)

        start = cover.d
        end = cover.end
        assert np.array_equal(np.array([start, end]), np.array([[0, -10], [38, 100]]))
