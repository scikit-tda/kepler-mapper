import km

# Load digits data
from sklearn import datasets
data, labels = datasets.load_digits().data, datasets.load_digits().target

# Create images for a custom tooltip array
import StringIO
from scipy.misc import imsave, toimage
import base64
tooltip_s = []
for image_data in data:
	output = StringIO.StringIO()
	img = toimage(image_data.reshape((8,8))) # Data was a flat row of 64 "pixels".
	img.save(output, format="PNG")
	contents = output.getvalue()
	tooltip_s.append( """ <img src="data:image/png;base64,%s"> """%base64.b64encode(contents).replace("\n","") )
	output.close()

tooltip_s = km.np.array(tooltip_s) # need to make sure to feed it as a NumPy array, not a list

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(cluster_algorithm=km.cluster.DBSCAN(eps=0.3, min_samples=15), 
						 reducer = km.manifold.TSNE(), nr_cubes=35, overlap_perc=0.9, 
						 link_local=False, verbose=2)

# Fit and transform data
data = mapper.fit_transform(data)

# Create the graph
complex = mapper.map(data, dimension_index=[0,1], dimension_name="t-SNE(2) 2D")

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)

# Tooltips with image data for every cluster member
mapper.visualize(complex, "keplermapper_digits_custom_tooltips.html", "Digits", graph_gravity=0.25, custom_tooltips=tooltip_s)
# Tooltips with the target y-labels for every cluster member
mapper.visualize(complex, "keplermapper_digits_ylabel_tooltips.html", "Digits", graph_gravity=0.25, custom_tooltips=labels)