import pytest
import numpy as np

from km import KeplerMapper


@pytest.fixture
def mapper():
    return KeplerMapper()

@pytest.fixture
def data():
    data = np.random.rand(100,10)
    return data


def test_lens_size(mapper, data):
    lens = mapper.fit_transform(data)

    assert lens.shape[0] == data.shape[0]

def test_num_cubes(mapper):
    nr_cubes = 10
    nr_dimensions = 2
    bins = mapper._cube_coordinates_all(nr_cubes, nr_dimensions)

    assert len(bins) == nr_cubes**nr_dimensions

def test_cube_dimensions(mapper):
    nr_cubes = 10
    nr_dimensions = 2
    bins = mapper._cube_coordinates_all(nr_cubes, nr_dimensions)

    assert len(bins[0]) == nr_dimensions
