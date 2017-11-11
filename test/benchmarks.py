"""
Build a rigorous benchmarking suite to compare all future changes.

For different parameter sets, I want to know
    * how long the construction takes
    * how much memory space it takes

Should provide summary of


"""


import cProfile, pstats, io



import time

import numpy as np
from sklearn import datasets

from kmapper import KeplerMapper


def profile():
    blob_list = []
    for i in range(100):
        data, _ = datasets.make_blobs(100)
        blob_list.append(data)

    mapper = KeplerMapper(verbose=0)

    pr = cProfile.Profile()
    pr.enable()

    for data in blob_list:
        lens = mapper.fit_transform(data)
        graph = mapper.map(lens,
                           data,
                           nr_cubes=10,
                           overlap_perc=0.2)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats("kmapper")
    print(s.getvalue())


if __name__ == "__main__":

    profile()
