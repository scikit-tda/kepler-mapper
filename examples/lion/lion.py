import numpy as np
import sklearn
import kmapper as km

data = np.genfromtxt('lion-reference.csv', delimiter=',')

mapper = km.KeplerMapper(verbose=1)

lens = mapper.fit_transform(data)

graph = mapper.map(lens,
                   data,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
                   cover=km.Cover(n_cubes=10, perc_overlap=0.2))

mapper.visualize(graph,
                 path_html="lion_keplermapper_output.html")

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
