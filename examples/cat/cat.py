import km

data = km.np.genfromtxt('cat-reference.csv',delimiter=',')

mapper = km.KeplerMapper(cluster_algorithm=km.cluster.DBSCAN(eps=0.1, min_samples=5), nr_cubes=10, overlap_perc=0.8, verbose=1)

mapper.fit(data)

complex = mapper.map(data, dimension_index=1, dimension_name="Y-axis")

mapper.visualize(complex, "cat_keplermapper_output.html", "cat-reference.csv")

# You may want to visualize the original point cloud data in 3D scatter too
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.savefig("cat-reference.csv.png")
plt.show()
"""