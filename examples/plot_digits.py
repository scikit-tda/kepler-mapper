"""

Digits Dataset
================

This digits example shows two ways of customizing the tooltips options in the HTML visualization. It generates the visualization with tooltips set as the y-label, or number of the image. The second generated result uses the actual image in the tooltips.

`Visualization with y-label tooltip <../../_static/digits_ylabel_tooltips.html>`_

`Visualization with custom tooltips <../../_static/digits_custom_tooltips.html>`_

"""

import io
import sys
import base64

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
import kmapper as km

try:
    from scipy.misc import imsave, toimage
except ImportError as e:
    print("imsave requires you to install pillow. Run `pip install pillow` and then try again.")
    sys.exit()


# Load digits dat
data, labels = datasets.load_digits().data, datasets.load_digits().target

# Create images for a custom tooltip array
tooltip_s = []
for image_data in data:
    output = io.BytesIO()
    img = toimage(image_data.reshape((8, 8)))  # Data was a flat row of 64 "pixels".
    img.save(output, format="PNG")
    contents = output.getvalue()
    img_encoded = base64.b64encode(contents)
    img_tag = """<img src="data:image/png;base64,{}">""".format(img_encoded.decode('utf-8'))
    tooltip_s.append(img_tag)
    output.close()

tooltip_s = np.array(tooltip_s)  # need to make sure to feed it as a NumPy array, not a list

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data,
                                      projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(projected_data,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                   cover=km.Cover(35, 0.4))

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html" )
# Tooltips with image data for every cluster member
mapper.visualize(graph,
                 title="Handwritten digits Mapper",
                 path_html="output/digits_custom_tooltips.html",
                 color_values=labels,
                 custom_tooltips=tooltip_s)
# Tooltips with the target y-labels for every cluster member
mapper.visualize(graph,
                 title="Handwritten digits Mapper",
                 path_html="output/digits_ylabel_tooltips.html",
                 custom_tooltips=labels)

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()
