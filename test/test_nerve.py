import pytest
from kmapper import GraphNerve


class TestNerve():
    def test_graphnerve(self):
        nerve = GraphNerve()

        groups = {"a": [1, 2, 3, 4], "b": [1, 2, 5], "c": [5, 6, 7]}
        links, simplices = nerve(groups)

        # all vertices are simplices
        assert all([[k] in simplices for k in groups])

        # graph should be a -- b -- c
        assert "c" not in links["a"] and "a" not in links["c"]
        assert "b" in links["a"] or "a" in links["b"]
        assert "c" in links["b"] or "b" in links["c"]


    def test_min_intersection(self):
        nerve = GraphNerve(min_intersection=2)

        groups = {"a": [1,2,3,4], "b": [1,2,5], "c": [5,6,7]}
        links, simplices = nerve(groups)

        # all vertices are simplices
        assert all([[k] in simplices for k in groups])

        # graph should be a -- b    c
        assert "c" not in links["a"] and "a" not in links["c"]
        assert "b" in links["a"] or "a" in links["b"]
        assert "c" not in links["b"] and "b" not in links["c"]


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
