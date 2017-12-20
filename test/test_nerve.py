import pytest
from kmapper import GraphNerve


class TestLinker():
    # TODO: eventually we will make linker its own class that will be able to
    #       construct general simplicial complexes and
    #       something suitable for computing persistent homology
    def test_finds_a_link(self):
        nerve = GraphNerve()
        groups = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
        links, _ = nerve(groups)

        assert "a" in links or "b" in links
        assert links["a"] == ["b"] or links["b"] == ["a"]

    def test_no_link(self):
        nerve = GraphNerve()
        groups = {"a": [1, 2, 3, 4], "b": [5, 6, 7]}

        links, _ = nerve(groups)
        assert not links

    def test_pass_through_result(self):
        nerve = GraphNerve()
        groups = {"a": [1], "b": [2]}

        res = dict()
        links, _ = nerve(groups)

        assert res == links
