from __future__ import division
import pytest
import numpy as np
from sklearn import datasets, preprocessing

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
            
    def test_perc_overlap(self, CoverClass):
        '''
        2 cubes with 50% overlap and a range of [0,1] should lead to two cubes with intervals:
            [0, .75]
            [.25, 1]
        '''
        
        data = np.array([ [0,0],
                          [1,.25],
                          [2,.5],
                          [3,.75],
                          [4,1]])
        
        cover = Cover(n_cubes=2, perc_overlap=0.5)
        cubes = cover.define_bins(data)
        cubes = list(cubes)
        entries = [cover.find_entries(data, cube) for cube in cubes]
        
        for i in (0,1,2,3):
            assert data[i] in entries[0]
        for i in (1,2,3,4):
            assert data[i] in entries[1]


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

        test_cases = [
            {
                "cubes": 1,
                "range": [0,4],
                "overlap": 0.4,
                "chunk": 4 * (1 + 0.4)
            },
            {
                "cubes": 1,
                "range": [0,4],
                "overlap": 2.4,
                "chunk": 4 * (1 + 2.4)
            },
            {
                "cubes": 2,
                "range": [-4, 4],
                "overlap": 0.5,
                "chunk": 6
            },
            {
                "cubes": 2,
                "range": [-4, 4],
                "overlap": 0.5,
                "chunk": 6
            },
            {
                "cubes": 10,
                "range": [-4, 4],
                "overlap": 0.5,
                "chunk": 1.2
            },
            {
                "cubes": 10,
                "range": [-4, 4],
                "overlap": 1.0,
                "chunk": 2.0
            }
        ]


        for test_case in test_cases:
            scaler = preprocessing.MinMaxScaler(
                feature_range=test_case['range']
            )
            data = scaler.fit_transform(np.arange(20).reshape(10, 2))

            cover = Cover(
                n_cubes=test_case['cubes'], 
                perc_overlap=test_case['overlap']
            )
            _ = cover.define_bins(data)
            assert cover.chunk_dist[0] == pytest.approx(test_case['chunk'])


    def test_equal_entries(self):
        settings = {
            "cubes": 10,
            "overlap": 0.5
        }

        # uniform data:
        data = np.arange(0,100)
        data = data[:,np.newaxis]
        lens = data

        cov = Cover(settings["cubes"], settings["overlap"])

        # Prefix'ing the data with an ID column
        ids = np.array([x for x in range(lens.shape[0])])
        lens = np.c_[ids, lens]

        bins = cov.define_bins(lens)

        bins = list(bins)  # extract list from generator

        assert len(bins) == settings["cubes"]

        cube_entries = [cov.find_entries(lens,cube) for cube in bins]

        for c1, c2 in list(zip(cube_entries, cube_entries[1:]))[2:]:
            c1, c2 = c1[:,0], c2[:,0] # indices only

            calced_overlap = len(set(list(c1)).intersection(set(list(c2)))) / max(len(c1), len(c2))
            assert calced_overlap == pytest.approx(0.5)

    def test_125_replication(self):
         # uniform data:
        data = np.arange(0,100)
        data = data[:,np.newaxis]
        lens = data

        cov = Cover(10, 0.5)

        # Prefix'ing the data with an ID column
        ids = np.array([x for x in range(lens.shape[0])])
        lens = np.c_[ids, lens]

        bins = cov.define_bins(lens)

        bins = list(bins)  # extract list from generator

        cube_entries = [cov.find_entries(lens,cube) for cube in bins]
 
        cubesizes = [len(c) for c in cube_entries]
        assert len(set(cubesizes)) == 1, "Each cube should have the same number of entries"

        overlaps = [len(set(list(c1[:,0])).intersection(set(list(c2[:,0])))) for c1, c2 in zip(cube_entries, cube_entries[1:])]
        assert len(set(overlaps)) == 1, "Each overlap should have the same number of entries. "

    def test_bound_is_min(self):
        data = np.arange(30).reshape(10, 3)
        cov = Cover(n_cubes=10)
        _ = cov.define_bins(data)
        bounds = list(zip(cov.d, range(1, 10)))
        assert all(b[0] == b[1] for b in bounds)

    def test_entries_in_correct_cubes(self):
        # TODO: this test is a little hacky

        data_vals = np.arange(20)
        data = np.zeros((20, 2))
        data[:, 0] = np.arange(20, dtype=int)  # Index row
        data[:, 1] = data_vals
    
        cover = Cover(n_cubes=10, perc_overlap=0.2)
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
