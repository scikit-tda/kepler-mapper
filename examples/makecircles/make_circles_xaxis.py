import km

# Make fairly noisy circles
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0])

# Create dictionary called 'complex' with nodes, edges and meta-information
complex = mapper.map(projected_X=projected_data, inverse_X=data, 
					 clusterer=km.cluster.DBSCAN(eps=0.1, min_samples=10), 
					 nr_cubes=20, overlap_perc=0.1)

# Visualize it
mapper.visualize(complex, path_html="keplermapper-makecircles-xaxis.html", 
				 title="datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)",
				 custom_tooltips=labels, color_function="average_signal_cluster")