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
    num_sets = 100
    blob_size = 1000
    nr_cubes = 10
    overlap = 0.2

    blob_list = []
    for i in range(num_sets):
        data, _ = datasets.make_blobs(blob_size)
        blob_list.append(data)

    mapper = KeplerMapper(verbose=0)

    pr = cProfile.Profile()
    pr.enable()

    for data in blob_list:
        lens = mapper.fit_transform(data)
        graph = mapper.map(lens, data, nr_cubes=nr_cubes, overlap_perc=overlap)

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats("kmapper")
    print(
        "Ran {} blobs of size {} with params (nr_cubes:{}\toverlap:{})".format(
            num_sets, blob_size, nr_cubes, overlap
        )
    )
    print(s.getvalue())


if __name__ == "__main__":
    profile()
