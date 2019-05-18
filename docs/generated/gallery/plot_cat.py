""" 

3D Cat Data
============


This example generates a Mapper built from a point-cloud sampled from a 3D model of a cat.

`Visualization of the cat mapper <../../_static/cat.html>`_


"""


import numpy as np
import sklearn
import kmapper as km

data = np.genfromtxt('data/cat-reference.csv', delimiter=',')

mapper = km.KeplerMapper(verbose=2)

lens = mapper.fit_transform(data)

graph = mapper.map(lens,
                   data,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
                   cover=km.Cover(n_cubes=15, perc_overlap=0.2))

mapper.visualize(graph,
                 path_html="output/cat.html")

km.draw_matplotlib(graph)

import matplotlib.pyplot as plt
plt.show()
