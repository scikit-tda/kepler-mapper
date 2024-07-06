"""

3D Cat Data
============


This example generates a Mapper built from a point-cloud sampled from a 3D model of a cat.

`Visualization of the cat mapper <../../_static/cat.html>`_


"""
# sphinx_gallery_thumbnail_path = '../examples/cat/cat-reference-viz.png'

import numpy as np
import sklearn
import kmapper as km
from pathlib import Path
import matplotlib.pyplot as plt

if Path("data/cat-reference.csv").exists():
    cat_path = "data/cat-reference.csv"
elif Path("cat-reference.csv").exists():
    cat_path = "cat-reference.csv"
else:
    raise FileNotFoundError
data = np.genfromtxt(cat_path, delimiter=",")

mapper = km.KeplerMapper(verbose=2)

lens = mapper.fit_transform(data)

graph = mapper.map(
    lens,
    data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
    cover=km.Cover(n_cubes=15, perc_overlap=0.2),
)

if Path("output/").is_dir():
    prepend = "output/"
else:
    prepend = "./"

mapper.visualize(graph, path_html=prepend + "cat.html")

km.draw_matplotlib(graph)


plt.show()
