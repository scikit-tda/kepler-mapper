from __future__ import division

from itertools import product
import numpy as np

# TODO: Incorporate @pablodecm's cover API.

class Cover():
    """Helper class that defines the default covering scheme

    functions
    ---------
    cubes:          @property, returns an iterable of all bins in the cover.
    find_entries:   Find all entries in the input data that are in the given cube.
    """

    def __init__(self, nr_cubes=10, overlap_perc=0.2):
        self.nr_cubes = nr_cubes
        self.overlap_perc = overlap_perc

    def define_bins(self, data):
        """
        Helper function to get origin coordinates for our intervals/hypercubes
        Useful for looping no matter the number of cubes or dimensions
        Example:   	if there are 4 cubes per dimension and 3 dimensions
                        return the bottom left (origin) coordinates of 64 hypercubes,
                        as a sorted list of Numpy arrays

        This function must assume that the first column of data are indices.
        """

        indexless_data = data[:, 1:]
        bounds = (np.min(indexless_data, axis=0),
                  np.max(indexless_data, axis=0))

        # We chop up the min-max column ranges into 'nr_cubes' parts
        self.chunk_dist = (bounds[1] - bounds[0]) / self.nr_cubes

        # We calculate the overlapping windows distance
        self.overlap_dist = self.overlap_perc * self.chunk_dist

        # We find our starting point
        self.d = bounds[0]

        # Use a dimension index array on the projected X
        # (For now this uses the entire dimensionality, but we keep for experimentation)
        self.di = np.array(range(1, data.shape[1]))
        self.nr_dimensions = len(self.di)

        if type(self.nr_cubes) is not list:
            cubes = [self.nr_cubes] * self.nr_dimensions
        else:
            assert len(self.nr_cubes) == self.nr_dimensions, "There are {} ({}) dimensions specified but {} dimensions needing specification. If you supply specific number of cubes for each dimension, please supply the correct number.".format(
                len(self.nr_cubes), self.nr_cubes, self.nr_dimensions)
            cubes = self.nr_cubes

        coordinates = map(np.asarray, product(
            *(range(i) for i in cubes)))

        return coordinates

    def find_entries(self, data, cube, verbose=0):
        """Find all entries in data that are in the given cube

        Input:      data, cube (an item from the list of cubes provided by `cover.cubes`)
        Output:     all entries in data that are in cube.
        """

        chunk = self.chunk_dist
        overlap = self.overlap_dist
        lower_bound = self.d + (cube * chunk)
        upper_bound = lower_bound + chunk + overlap

        # Slice the hypercube
        entries = (data[:, self.di] >= lower_bound) & \
                  (data[:, self.di] < upper_bound)

        hypercube = data[np.invert(np.any(entries == False, axis=1))]

        return hypercube
