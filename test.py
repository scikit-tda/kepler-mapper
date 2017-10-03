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
