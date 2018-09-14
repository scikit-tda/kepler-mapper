"""
    These are tests that ensure changes to the API don't break backwards compatibility, for at least a few releases.

"""

import pytest

import warnings
import numpy as np
from kmapper import KeplerMapper, Cover


class TestAPIMaintenance:
    """ These tests just confirm that new api changes are backwards compatible"""

    def test_nr_cubes_cover(self):
        mapper = KeplerMapper()

        with pytest.deprecated_call():
            cover = Cover(nr_cubes=17)  # strange number

        assert cover.n_cubes == 17

    def test_overlap_perc_cover(self):
        mapper = KeplerMapper()

        with pytest.deprecated_call():
            cover = Cover(overlap_perc=17)  # strange number

        assert cover.perc_overlap == 17

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
        c2 = mapper.cover

        assert c2.perc_overlap == 0.1

        _ = mapper.map(lens, data, overlap_perc=0.1)
        c2 = mapper.cover

        assert c2.n_cubes == 10

    def test_no_warn_normally(self, recwarn):
        """ Confirm that deprecation warnings behave as expected"""
        mapper = KeplerMapper()
        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        warnings.simplefilter("always")
        graph = mapper.map(lens, data)

        assert len(recwarn) == 0
        assert DeprecationWarning not in recwarn
