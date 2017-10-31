# Generate a random sampling of a 2d donut.



import random
import math
import json


import matplotlib.pyplot as plt


random.seed(0)

NUM_POINTS = 2000
MIDDLE_RADIUS = 10
BAND_WIDTH = 5

# Method from Wolfram http://mathworld.wolfram.com/SpherePointPicking.html

U = [random.random() for _ in range(NUM_POINTS)]
V = [random.random() for _ in range(NUM_POINTS)]
R = [random.random() for _ in range(NUM_POINTS)]

U = [1 for _ in range(100)]
V = [0.01* i for i in range(100)]

thetas = [2*math.pi*u for u in U]
phis = [math.acos(2*v - 1) for v in V]

xs = [math.cos(t)*math.sin(p) for t, p, r in zip(thetas, phis, R)]
ys = [math.sin(t)*math.cos(p) for t, p, r in zip(thetas, phis, R)]
zs = [math.cos(p) for p, r in zip(phis, R)]
points = list(zip(xs, ys, zs))


plt.scatter(thetas, phis)
plt.show()



with open('sphere.json', 'w') as outfile:
    json.dump(points, outfile)
