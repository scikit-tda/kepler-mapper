import pytest
import numpy as np
from kmapper import KeplerMapper

@pytest.fixture
def sc():
    X = np.random.random((200, 5))
    km = KeplerMapper()
    lens = km.fit_transform(X)
    sc = km.map(lens, X)
    return sc
