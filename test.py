import pytest
import numpy as np

from km import KeplerMapper


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
