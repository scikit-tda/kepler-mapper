# Generate a random sampling of a 2d donut.



import random
import math
import json

random.seed(0)

NUM_POINTS = 2000
MIDDLE_RADIUS = 10
BAND_WIDTH = 5

uniform_position = [random.random() for _ in range(NUM_POINTS)]
uniform_radius = [random.random() for _ in range(NUM_POINTS)]


def radius(u):
    return u*BAND_WIDTH+MIDDLE_RADIUS

xs = [radius(r) * math.cos(2*math.pi*p) for r, p in zip(uniform_position, uniform_radius)]
ys = [radius(r) * math.sin(2*math.pi*p) for r, p in zip(uniform_position, uniform_radius)]

points = list(zip(xs, ys))

with open('torus.json', 'w') as outfile:
    json.dump(points, outfile)
