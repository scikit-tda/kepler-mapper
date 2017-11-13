import json
import itertools

import kmapper as km

import numpy as np
from scipy.spatial import ConvexHull

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Accent6
from bokeh.models import ColumnDataSource

# Get data
with open('demo/data/torus.json', 'rb') as file:
    j = json.load(file)

data = np.array(j)
xs = [d[0] for d in data]
ys = [d[1] for d in data]

mapper = km.KeplerMapper(verbose=10)

lens = mapper.fit_transform(data, projection=[1])

def make_1d_uniform(lens):
    ''' To test whether variable bucket mapper helps,
         we will lay the lens out as uniform as possible. '''

    assert lens.shape[1] == 1, "Lens must be 1 dimensional for this uniformation to work"

    # find uniform points on interval [0,1]
    uniform_points = (1/(lens.shape[0])) * np.arange(lens.shape[0])

    # get ids for each row
    ids = np.array(range(lens.shape[0]))[np.newaxis].T
    ided = np.concatenate((ids, lens), axis=1)

    # sort each row
    srtd = np.array(sorted(ided, key=lambda x: x[1]))

    # replace points with uniform points
    srtd[:,1] = uniform_points

    # sort back, because indices matter
    lens = np.array(sorted(srtd, key=lambda x: x[0]))[:,1][np.newaxis].T

    return lens

#lens = make_1d_uniform(lens)

graph = mapper.map(lens,
                   data,
                   clusterer=km.cluster.DBSCAN(eps=2, min_samples=10),
                   nr_cubes=30,
                   overlap_perc=0.4)

patches_x = []
patches_y = []

patches = dict()
patches["xs"] = []
patches["ys"] = []
patches["group"] = []

# import pdb; pdb.set_trace()

for key, value in list(dict(graph["nodes"]).items()):
    bucket = int(key.split("_")[0])
    points = np.take(data, value, axis=0)
    hull = np.take(points, ConvexHull(points).vertices, axis=0)

    #import pdb; pdb.set_trace()
    patches["xs"].append(hull[:,0])
    patches["ys"].append(hull[:,1])
    patches["group"].append(bucket)


total_bins = len(set(patches["group"]))
color_choices = itertools.cycle(Accent6)

# import pdb; pdb.set_trace()

colormap = dict(zip(set(patches["group"]), color_choices))
colors = [colormap[x] for x in patches['group']]
patches["colors"] = colors

p = figure(plot_width=800, plot_height=800)
p.circle(y=ys, x=xs, size=2.5, color="black", alpha=1)
p.patches(xs="xs", ys="ys", color="colors", alpha=0.5, source=patches)#, color=colors)
show(p)

mapper.visualize(graph,
                 path_html="demo/keplermapper_output.html")



# plt.scatter(xs,ys)
# plt.show()

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
