import kmapper as km

# Make fairly noisy circles
import sklearn
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
lens = mapper.fit_transform(data, projection=[0])

# Create dictionary called 'simplicial_complex' with nodes, edges and meta-information
simplicial_complex = mapper.map(lens, X=data,
                                clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=10),
                                cover=km.Cover(n_cubes=20, perc_overlap=0.1))

# Visualize it
mapper.visualize(simplicial_complex, path_html="keplermapper-makecircles-xaxis.html",
                 custom_meta={'Data': "datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)"},
                 custom_tooltips=labels)
