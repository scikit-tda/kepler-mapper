"""

3D Horse Data
================


This example generates a Mapper built from a point-cloud sampled from a 3D model of a horse.

`Visualization of the horse data <../../_static/horse.html>`_



"""

# sphinx_gallery_thumbnail_path = '../examples/horse/horse-reference.png'
import matplotlib.pyplot as plt
import numpy as np
import sklearn

import kmapper as km
from pathlib import Path

if Path("data/horse-reference.csv").exists():
    horse_path = "data/horse-reference.csv"
elif Path("horse-reference.csv").exists():
    horse_path = "horse-reference.csv"
else:
    raise FileNotFoundError

data = np.genfromtxt(horse_path, delimiter=",")

mapper = km.KeplerMapper(verbose=2)


lens = mapper.fit_transform(data)


graph = mapper.map(
    lens,
    data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
    cover=km.Cover(30, 0.2),
)

if Path("output/").is_dir():
    prepend = "output/"
else:
    prepend = "./"

mapper.visualize(
    graph, path_html=prepend + "horse.html", custom_tooltips=np.arange(len(lens))
)


km.drawing.draw_matplotlib(graph)
plt.show()
