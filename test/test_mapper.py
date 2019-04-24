import pytest
import numpy as np

from collections import defaultdict

import warnings
from kmapper import KeplerMapper, Cover, cluster

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.manifold import MDS
from scipy import sparse
from scipy.spatial import distance
from sklearn import neighbors


class TestLogging:
    """ Simple tests that confirm map completes at each logging level
    """

    def test_runs_with_logging_0(self, capsys):
        mapper = KeplerMapper(verbose=0)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

        captured = capsys.readouterr()
        assert captured[0] == ""

    def test_runs_with_logging_1(self):
        mapper = KeplerMapper(verbose=1)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

    def test_runs_with_logging_2(self):
        mapper = KeplerMapper(verbose=2)
        data = np.random.rand(100, 2)
        graph = mapper.map(data)

    def test_logging_in_project(self, capsys):
        mapper = KeplerMapper(verbose=2)
        data = np.random.rand(100, 2)
        lens = mapper.project(data)

        captured = capsys.readouterr()
        assert "Projecting on" in captured[0]

    def test_logging_in_fit_transform(self, capsys):
        mapper = KeplerMapper(verbose=2)
        data = np.random.rand(100, 2)
        lens = mapper.fit_transform(data)

        captured = capsys.readouterr()
        assert "Composing projection pipeline of length 1" in captured[0]


class TestDataAccess:
    def test_members_from_id(self):
        mapper = KeplerMapper(verbose=1)
        data = np.random.rand(100, 2)

        ids = np.random.choice(10, 100)
        data[ids] = 2

        graph = mapper.map(data)
        graph["nodes"]["new node"] = ids
        mems = mapper.data_from_cluster_id("new node", graph, data)
        np.testing.assert_array_equal(data[ids], mems)

    def test_wrong_id(self):
        mapper = KeplerMapper(verbose=1)
        data = np.random.rand(100, 2)

        graph = mapper.map(data)
        mems = mapper.data_from_cluster_id("new node", graph, data)
        np.testing.assert_array_equal(mems, np.array([]))


class TestMap:
    def test_simplices(self):
        mapper = KeplerMapper()

        X = np.random.rand(100, 2)
        lens = mapper.fit_transform(X)
        graph = mapper.map(
            lens,
            X=X,
            cover=Cover(n_cubes=3, perc_overlap=0.75),
            clusterer=cluster.DBSCAN(metric="euclidean", min_samples=3),
        )
        assert max([len(s) for s in graph["simplices"]]) <= 2

        nodes = [n for n in graph["simplices"] if len(n) == 1]
        edges = [n for n in graph["simplices"] if len(n) == 2]
        assert len(nodes) == 3
        assert len(edges) == 3

    def test_precomputed(self):
        mapper = KeplerMapper()

        X = np.random.rand(100, 2)
        X_pdist = distance.squareform(distance.pdist(X, metric="euclidean"))

        lens = mapper.fit_transform(X_pdist)

        graph = mapper.map(
            lens,
            X=X_pdist,
            cover=Cover(n_cubes=10, perc_overlap=0.8),
            clusterer=cluster.DBSCAN(metric="precomputed", min_samples=3),
            precomputed=True,
        )
        graph2 = mapper.map(
            lens,
            X=X,
            cover=Cover(n_cubes=10, perc_overlap=0.8),
            clusterer=cluster.DBSCAN(metric="euclidean", min_samples=3),
        )

        assert graph["links"] == graph2["links"]
        assert graph["nodes"] == graph2["nodes"]
        assert graph["simplices"] == graph2["simplices"]

    def test_remove_duplicates_argument(self, capsys):
        mapper = KeplerMapper(verbose=1)
        X = np.random.rand(100, 5)

        lens = mapper.project(X)
        graph = mapper.map(
            lens,
            X=X,
            cover=Cover(n_cubes=2, perc_overlap=1),
            clusterer=cluster.DBSCAN(metric="euclidean", min_samples=3),
            remove_duplicate_nodes=True,
        )

        captured = capsys.readouterr()
        assert "duplicate nodes" in captured[0]

    def test_lots_of_repeats(self, capsys):
        nodes = defaultdict(list)
        for i in range(4):
            nodes["cube{}_cluster0".format(i)] = [0, 1]
        for i in range(3):
            nodes["cube{}_cluster1".format(i)] = [1, 2]
        nodes["cube1_cluster2"] = [1, 2, 3]
        nodes["cube2_cluster3"] = [2, 3, 4]

        deduped_nodes = KeplerMapper(verbose=1)._remove_duplicate_nodes(nodes)
        assert len(deduped_nodes) == 4

        captured = capsys.readouterr()
        assert "Merged 5 duplicate nodes." in captured[0]

    def test_no_duplicates(self):
        nodes = defaultdict(list)
        nodes["cube1_cluster0"] = [0]
        nodes["cube1_cluster1"] = [1, 2]
        nodes["cube1_cluster2"] = [1, 2, 3]
        nodes["cube2_cluster1"] = [2, 3, 4]

        size = len(nodes)
        deduped_nodes = KeplerMapper()._remove_duplicate_nodes(nodes)
        assert len(deduped_nodes) == len(nodes)
        assert len(deduped_nodes) == size

    def test_duplicates_purity(self):
        nodes = defaultdict(list)
        nodes["cube1_cluster0"] = [0]
        nodes["cube1_cluster1"] = [1, 2]
        nodes["cube1_cluster2"] = [1, 2, 3]
        nodes["cube2_cluster1"] = [2, 3, 4]

        deduped_nodes = KeplerMapper()._remove_duplicate_nodes(nodes)
        assert deduped_nodes["cube2_cluster1"] == [2, 3, 4]

        nodes["cube2_cluster1"].append(5)
        assert deduped_nodes["cube2_cluster1"] == [2, 3, 4]

    def test_remove_duplicates_direct(self):
        nodes = defaultdict(list)
        nodes["cube1_cluster0"] = [0]
        nodes["cube1_cluster1"] = [1, 2]
        nodes["cube1_cluster2"] = [1, 2, 3]
        nodes["cube2_cluster0"] = [1, 2]
        nodes["cube2_cluster1"] = [2, 3, 4]

        mapper = KeplerMapper()
        deduped_nodes = mapper._remove_duplicate_nodes(nodes)

        assert len(deduped_nodes) < len(nodes)
        assert len(deduped_nodes) == 4
        assert (
            "cube1_cluster1|cube2_cluster0" in deduped_nodes
            or "cube2_cluster0|cube1_cluster1" in deduped_nodes
        )

    def test_precomputed_with_knn_lens(self):
        mapper = KeplerMapper()

        X = np.random.rand(100, 5)

        lens = mapper.fit_transform(
            X, projection="knn_distance_3", distance_matrix="chebyshev"
        )
        assert lens.shape == (100, 1)

    def test_affinity_prop_clustering(self):
        mapper = KeplerMapper()

        X = np.random.rand(100, 2)
        lens = mapper.fit_transform(X)

        graph = mapper.map(lens, X, clusterer=cluster.AffinityPropagation())


class TestLens:
    # TODO: most of these tests only accommodate the default option. They need to be extended to incorporate all possible transforms.

    # one test for each option supported
    def test_str_options(self):
        mapper = KeplerMapper()

        data = np.random.rand(100, 10)

        options = [
            ["sum", np.sum],
            ["mean", np.mean],
            ["median", np.median],
            ["max", np.max],
            ["min", np.min],
            ["std", np.std],
            ["l2norm", np.linalg.norm],
        ]

        first_point = data[0]
        last_point = data[-1]
        for tag, func in options:
            lens = mapper.fit_transform(data, projection=tag, scaler=None)
            np.testing.assert_almost_equal(lens[0][0], func(first_point))
            np.testing.assert_almost_equal(lens[-1][0], func(last_point))

        # For dist_mean, just make sure the code runs without breaking, not sure how to test this best
        lens = mapper.fit_transform(data, projection="dist_mean", scaler=None)

    def test_knn_distance(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.project(data, projection="knn_distance_4", scaler=None)

        nn = neighbors.NearestNeighbors(n_neighbors=4)
        nn.fit(data)
        lens_confirm = np.sum(
            nn.kneighbors(data, n_neighbors=4, return_distance=True)[0], axis=1
        ).reshape((-1, 1))

        assert lens.shape == (100, 1)
        np.testing.assert_array_equal(lens, lens_confirm)

    def test_distance_matrix(self):
        # todo, test other distance_matrix functions
        mapper = KeplerMapper(verbose=4)
        X = np.random.rand(100, 10)
        lens = mapper.fit_transform(X, distance_matrix="euclidean")

        X_pdist = distance.squareform(distance.pdist(X, metric="euclidean"))
        lens2 = mapper.fit_transform(X_pdist)

        np.testing.assert_array_equal(lens, lens2)

    def test_sparse_array(self):
        mapper = KeplerMapper()

        data = sparse.random(100, 10)
        lens = mapper.fit_transform(data)

    def test_lens_size(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.fit_transform(data)

        assert lens.shape[0] == data.shape[0]

    def test_lens_names(self):
        mapper = KeplerMapper()
        data = np.random.rand(100, 5)
        lens = mapper.fit_transform(data)
        graph = mapper.map(lens, data)
        template = mapper.visualize(graph=graph)
        template2 = mapper.visualize(graph=graph, lens_names=[])

        assert template == template2

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
        lens = mapper.project(data, projection=PCA(n_components=1), scaler=None)

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
        np.testing.assert_array_equal(lens, lasso.predict(data).reshape((100, 1)))

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

        mapper = KeplerMapper(verbose=1)
        data = np.random.rand(100, 5)
        lens = mapper.project(data, projection=[0, 1])
        np.testing.assert_allclose(lens, data[:, :2], atol=atol)

        lens = mapper.project(data, projection=[0])
        np.testing.assert_allclose(lens, data[:, :1], atol=atol)

    def test_pipeline(self):
        # TODO: break this test into many smaller ones.
        input_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float64)
        atol_big = 0.1
        atol_small = 0.001

        mapper = KeplerMapper()

        lens_1 = mapper.fit_transform(
            input_data, projection=[[0, 1], "sum"], scaler=None
        )
        expected_output_1 = np.array([[3], [7], [11], [15]])

        lens_2 = mapper.fit_transform(input_data, projection=[[0, 1], "sum"])
        expected_output_2 = np.array([[0], [0.33], [0.66], [1.0]])

        lens_3 = mapper.fit_transform(
            input_data, projection=[[0, 1], "mean"], scaler=None
        )
        expected_output_3 = np.array([[1.5], [3.5], [5.5], [7.5]])

        lens_4 = mapper.fit_transform(input_data, projection=[[1], "mean"], scaler=None)
        expected_output_4 = np.array([[2], [4], [6], [8]])

        lens_5 = mapper.fit_transform(
            input_data,
            projection=[[0, 1], "l2norm"],
            scaler=None,
            distance_matrix=[False, "pearson"],
        )
        expected_output_5 = np.array([[2.236], [5.0], [7.81], [10.630]])

        lens_6 = mapper.fit_transform(
            input_data,
            projection=[[0, 1], [0, 1]],
            scaler=None,
            distance_matrix=[False, "cosine"],
        )
        expected_output_6 = np.array(
            [[0.0, 0.016], [0.016, 0.0], [0.026, 0.0013], [0.032, 0.0028]]
        )

        lens_7 = mapper.fit_transform(
            input_data,
            projection=[[0, 1], "l2norm"],
            scaler=None,
            distance_matrix=[False, "cosine"],
        )
        expected_output_7 = np.array([[0.044894], [0.01643], [0.026617], [0.032508]])

        lens_8 = mapper.fit_transform(input_data, projection=[[0, 1], "sum"])
        lens_9 = mapper.fit_transform(input_data, projection="sum")

        lens_10 = mapper.fit_transform(
            input_data, projection="sum", scaler=StandardScaler()
        )
        lens_11 = mapper.fit_transform(
            input_data, projection=[[0, 1], "sum"], scaler=[None, StandardScaler()]
        )
        expected_output_10 = np.array(
            [[-1.341641], [-0.447214], [0.447214], [1.341641]]
        )

        np.testing.assert_array_equal(lens_1, expected_output_1)
        np.testing.assert_allclose(lens_2, expected_output_2, atol=atol_big)
        np.testing.assert_array_equal(lens_3, expected_output_3)
        np.testing.assert_array_equal(lens_4, expected_output_4)
        np.testing.assert_allclose(lens_5, expected_output_5, atol=atol_small)
        np.testing.assert_allclose(lens_6, expected_output_6, atol=atol_small)
        np.testing.assert_allclose(lens_7, expected_output_7, atol=atol_small)
        np.testing.assert_allclose(lens_8, lens_9, atol=atol_small)
        np.testing.assert_allclose(lens_10, expected_output_10, atol=atol_small)
        np.testing.assert_array_equal(lens_10, lens_11)
        assert not np.array_equal(lens_10, lens_2)
        assert not np.array_equal(lens_10, lens_1)

    def test_map_sparse(self):
        mapper = KeplerMapper()

        data = sparse.random(100, 10, random_state=100101)
        lens = mapper.fit_transform(data)
        mapping = mapper.map(lens, data)

