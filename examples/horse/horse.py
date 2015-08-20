import km

data = km.np.genfromtxt('horse-reference.csv',delimiter=',')

mapper = km.KeplerMapper(cluster_algorithm=km.cluster.DBSCAN(eps=0.3, min_samples=3), nr_cubes=25, link_local=False, overlap_perc=0.7, verbose=1)

mapper.fit(data)

complex = mapper.map(data, dimension_index=1, dimension_name="Y-axis")

mapper.visualize(complex, "horse_keplermapper_output.html", "horse-reference.csv")

# You may want to visualize the original point cloud data in 3D scatter too
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.savefig("horse-reference.csv.png")
plt.show()
"""