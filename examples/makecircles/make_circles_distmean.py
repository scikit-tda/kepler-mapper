import kmapper as km

# Make very noisy circles
import sklearn
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=2)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection="dist_mean")

# Create dictionary called 'complex' with nodes, edges and meta-information
complex = mapper.map(projected_X=projected_data, inverse_X=data,
					 clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
					 nr_cubes=30, overlap_perc=0.2)

# Visualize it
mapper.visualize(complex, path_html="keplermapper-makecircles-distmean.html",
				 title="datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)",
				 custom_tooltips=labels, color_function="average_signal_cluster",
				 graph_gravity=0.03, graph_link_distance=30, graph_charge=-80)
