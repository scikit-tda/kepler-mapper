import pytest
import numpy as np

import warnings
from kmapper import KeplerMapper

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso


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

    def test_project_sklearn_class(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.project(data, projection=PCA(
            n_components=1), scaler=None)

        pca = PCA(n_components=1)
        lens_confirm = pca.fit_transform(data)
        assert lens.shape == (100, 1)
        np.testing.assert_array_equal(lens, lens_confirm)

    def test_tuple_projection(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        y = np.random.rand(100, 1)
        lasso = Lasso()
        lasso.fit(data, y)
        lens = mapper.project(data, projection=(lasso, data), scaler=None)

        # hard to test this, at least it doesn't fail
        assert lens.shape == (100, 1)
        np.testing.assert_array_equal(
            lens, lasso.predict(data).reshape((100, 1)))

    def test_tuple_projection_fit(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        y = np.random.rand(100, 1)
        lens = mapper.project(data, projection=(Lasso(), data, y), scaler=None)

        # hard to test this, at least it doesn't fail
        assert lens.shape == (100, 1)

    def test_projection_without_pipeline(self):
        # accomodate scaling, values are in (0,1), but will be scaled slightly
        atol = 0.1

        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.project(data, projection=[0, 1])
        np.testing.assert_allclose(lens, data[:, :2], atol=atol)

        lens = mapper.project(data, projection=[0])
        np.testing.assert_allclose(lens, data[:, :1], atol=atol)

    def test_pipeline(self):
        input_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        atol_big = 0.1
        atol_small = 0.001

        mapper = KeplerMapper()

        lens_1 = mapper.fit_transform(input_data,
                                      projection=[[0, 1], "sum"],
                                      scaler=None)
        expected_output_1 = np.array([[3], [7], [11], [15]])

        lens_2 = mapper.fit_transform(input_data,
                                      projection=[[0, 1], "sum"])
        expected_output_2 = np.array([[0], [0.33], [0.66], [1.]])

        lens_3 = mapper.fit_transform(input_data,
                                      projection=[[0, 1], "mean"],
                                      scaler=None)
        expected_output_3 = np.array([[1.5], [3.5], [5.5], [7.5]])

        lens_4 = mapper.fit_transform(input_data,
                                      projection=[[1], "mean"],
                                      scaler=None)
        expected_output_4 = np.array([[2], [4], [6], [8]])

        lens_5 = mapper.fit_transform(input_data,
                                      projection=[[0, 1], "l2norm"],
                                      scaler=None,
                                      distance_matrix=[False, "pearson"])
        expected_output_5 = np.array([[2.236], [5.], [7.81], [10.630]])

        lens_6 = mapper.fit_transform(input_data,
                                      projection=[[0, 1], [0, 1]],
                                      scaler=None,
                                      distance_matrix=[False, "cosine"])
        expected_output_6 = np.array([[0., 0.016],
                                      [0.016, 0.], [0.026, 0.0013], [0.032, 0.0028]])

        lens_7 = mapper.fit_transform(input_data,
                                      projection=[[0, 1], "l2norm"],
                                      scaler=None,
                                      distance_matrix=[False, "cosine"])
        expected_output_7 = np.array(
            [[0.044894], [0.01643], [0.026617], [0.032508]])

        lens_8 = mapper.fit_transform(input_data, projection=[[0, 1], "sum"])
        lens_9 = mapper.fit_transform(input_data, projection="sum")

        lens_10 = mapper.fit_transform(input_data, projection="sum",
                                       scaler=StandardScaler())
        lens_11 = mapper.fit_transform(input_data, projection=[[0, 1], "sum"],
                                       scaler=[None, StandardScaler()])
        expected_output_10 = np.array(
            [[-1.341641], [-0.447214], [0.447214], [1.341641]])

        np.testing.assert_array_equal(lens_1, expected_output_1)
        np.testing.assert_allclose(lens_2, expected_output_2, atol=atol_big)
        np.testing.assert_array_equal(lens_3, expected_output_3)
        np.testing.assert_array_equal(lens_4, expected_output_4)
        np.testing.assert_allclose(lens_5, expected_output_5, atol=atol_small)
        np.testing.assert_allclose(lens_6, expected_output_6, atol=atol_small)
        np.testing.assert_allclose(lens_7, expected_output_7, atol=atol_small)
        np.testing.assert_allclose(lens_8, lens_9, atol=atol_small)
        np.testing.assert_allclose(
            lens_10, expected_output_10, atol=atol_small)
        np.testing.assert_array_equal(lens_10, lens_11)
        assert not np.array_equal(lens_10, lens_2)
        assert not np.array_equal(lens_10, lens_1)


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

        assert c2.n_cubes == 10

    def test_no_warn_normally(self, recwarn):
        """ Confirm that deprecation warnings behave as expected"""
        mapper = KeplerMapper()
        data = np.random.rand(100, 10)
        lens = mapper.fit_transform(data)

        warnings.simplefilter('always')
        graph = mapper.map(lens, data)

        assert len(recwarn) == 0
        assert DeprecationWarning not in recwarn
