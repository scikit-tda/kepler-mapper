import pytest
import numpy as np

import warnings
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


class TestLens():
    # TODO: most of these tests only accomodate the default option. They need to be extended to incorporate all possible transforms.

    # one test for each option supported
    def test_str_options(self):
        mapper = KeplerMapper()

        data = np.random.rand(100, 10)

        options = [
            ['sum', np.sum],
            ['mean', np.mean],
            ['median', np.median],
            ['max', np.max],
            ['min', np.min],
            ['std', np.std],
            ['l2norm', np.linalg.norm]
        ]

        first_point = data[0]
        last_point = data[-1]
        for tag, func in options:
            lens = mapper.fit_transform(data, projection=tag, scaler=None)
            np.testing.assert_almost_equal(lens[0][0], func(first_point))
            np.testing.assert_almost_equal(lens[-1][0], func(last_point))

    def test_lens_size(self):
        mapper = KeplerMapper()

        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        assert lens.shape[0] == data.shape[0]

    def test_map_custom_lens(self):
        # I think that map currently requires fit_transform to be called first
        mapper = KeplerMapper()
        data = np.random.rand(100, 2)
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


class TestAPIMaintenance():
    """ These tests just confirm that new api changes are backwards compatible"""

    def test_warn_old_api(self):
        """ Confirm old api works but throws warning """

        mapper = KeplerMapper()
        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        with pytest.deprecated_call():
            graph = mapper.map(lens, data, nr_cubes=10)

        with pytest.deprecated_call():
            graph = mapper.map(lens, data, overlap_perc=10)

        with pytest.deprecated_call():
            graph = mapper.map(lens, data, nr_cubes=10, overlap_perc=0.1)

    def test_new_api_old_defaults(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        _ = mapper.map(lens, data, nr_cubes=10)
        c2 = mapper.coverer

        assert c2.overlap_perc == 0.1

        _ = mapper.map(lens, data, overlap_perc=0.1)
        c2 = mapper.coverer

        assert c2.nr_cubes == 10

    def test_no_warn_normally(self, recwarn):
        """ Confirm that deprecation warnings behave as expected"""
        mapper = KeplerMapper()
        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        warnings.simplefilter('always')
        graph = mapper.map(lens, data)

        assert len(recwarn) == 0
        assert DeprecationWarning not in recwarn
