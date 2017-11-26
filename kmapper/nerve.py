import itertools
from collections import defaultdict


class Nerve:
    """
    Base class for implementations of a nerve finder to build a Mapper complex.

    functions
    ---------
    __call__:     Return all simplices found by the nerve finder
    """
    def __init__(self):
        pass

    def __call__(self, nodes, links):
        raise NotImplementedError()


class GraphNerve(Nerve):
    """
    Creates the 1-skeleton of the Mapper complex.
    """
    def __call__(self, nodes, links):
        """
            Helper function to find edges of the overlapping clusters.

            TODO: generalize to take nerve.
        """
        if links is None:
            result = defaultdict(list)
        else:
            result = links

        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            if len(nodes[candidate[0]] + nodes[candidate[1]]) != len(set(nodes[candidate[0]] + nodes[candidate[1]])):
                result[candidate[0]].append(candidate[1])

        return result, [[n] for n in nodes] + [[x] + result[x] for x in result]
