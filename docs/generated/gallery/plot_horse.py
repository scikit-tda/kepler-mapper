"""

3D Horse Data
================


This example generates a Mapper built from a point-cloud sampled from a 3D model of a horse.

`Visualization of the horse data <../../_static/horse.html>`_



"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn

import kmapper as km

data = np.genfromtxt('data/horse-reference.csv', delimiter=',')

mapper = km.KeplerMapper(verbose=2)


lens = mapper.fit_transform(data)


graph = mapper.map(lens,
                   data,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
                   cover=km.Cover(30, 0.2))

mapper.visualize(graph,
                 path_html="output/horse.html",
                 custom_tooltips=np.arange(len(lens)))


km.drawing.draw_matplotlib(graph)
plt.show()

