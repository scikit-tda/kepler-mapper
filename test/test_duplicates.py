import pytest

from kmapper import KeplerMapper

from collections import defaultdict

@pytest.fixture
def with_duplicates():
    with_duplicates = defaultdict(list)
    with_duplicates["node1"] = [1,2,3]
    with_duplicates["node2"] = [2,3,4]
    with_duplicates["node3"] = [1,2,3]
    with_duplicates["node4"] = [9,8,7]

    return with_duplicates

def test_duplicates(with_duplicates):

    km = KeplerMapper()

    no_duplicates = km._remove_duplicates(with_duplicates)
    assert len(no_duplicates) < len(with_duplicates)

    # replicate structure of nodes
   
