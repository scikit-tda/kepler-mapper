import km
import numpy as np
import json

import matplotlib.pyplot as plt
import pprint

with open('sphere.json', 'rb') as file:
    j = json.load(file)

data = np.array(j)

print(data[0])

mapper = km.KeplerMapper(verbose=10)

lens = mapper.fit_transform(data, projection=[1])

graph = mapper.map(lens,
                   data,
                   clusterer=km.cluster.DBSCAN(eps=0.8, min_samples=5),
                   nr_cubes=10,
                   overlap_perc=0.3)

mapper.visualize(graph,
                 path_html="keplermapper_output.html")


xs = [d[0] for d in data]
ys = [d[1] for d in data]
zs = [d[2] for d in data]
plt.scatter(zs,ys)
plt.show()

# You may want to visualize the original point cloud data in 3D scatter too
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.savefig("lion-reference.csv.png")
plt.show()
"""
