import kmapper as km

# Make very noisy circles
import sklearn
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=2)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection="dist_mean")

# Create dictionary called 'simplicial_complex' with nodes, edges and meta-information
simplicial_complex = mapper.map(projected_data, X=data,
                                clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
                                cover=km.Cover(perc_overlap=0.2))

# Visualize it
mapper.visualize(simplicial_complex, path_html="keplermapper-makecircles-distmean.html",
                 custom_meta={"Data:": "datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)"},
                 custom_tooltips=labels,
                 color_values=labels)
