# Generate a random sampling of a 2d donut.

import random
import math
import json

import matplotlib.pyplot as plt


def torus(mean_radius=10, band_width=3, center=(0,0), points=1000):
    random.seed(0)

    uniform_position = [random.random() for _ in range(points)]
    uniform_radius = [random.random() for _ in range(points)]

    def radius(u):
        return u * band_width + mean_radius

    xs = [radius(r) * math.cos(2*math.pi*p) + center[0] for r, p in zip(uniform_position, uniform_radius)]
    ys = [radius(r) * math.sin(2*math.pi*p) + center[1] for r, p in zip(uniform_position, uniform_radius)]

    points = list(zip(xs, ys))

    return points


def badshape():
    """ create a shape that has homological features at multiple levels.
    """

    pass


if __name__ == "__main__":
    print("Make data")

    data = torus(10,2) + torus(5,0.5, (3,3)) #+ torus(2, 0.2, (10,5), 100)
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]

    with open('demo/data/torus.json', 'w') as outfile:
        json.dump(data, outfile)
