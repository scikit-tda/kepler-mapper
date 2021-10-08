from __future__ import division

try:
    from collections.abc import Iterable
except:
    from collections import Iterable

import warnings
from itertools import product
import numpy as np

# TODO: Incorporate @pablodecm's cover API.


__all__ = ["Cover", "CubicalCover"]


class Cover:
    """Helper class that defines the default covering scheme

    It calculates the cover based on the following formula for overlap. (https://arxiv.org/pdf/1706.00204.pdf)

    ::

                     |cube[i] intersection cube[i+1]|
        overlap = --------------------------------------
                              |cube[i]|


    Parameters
    ============

    n_cubes: int
        Number of hypercubes along each dimension. Sometimes referred to as resolution.

    perc_overlap: float
        Amount of overlap between adjacent cubes calculated only along 1 dimension.

    limits: Numpy Array (n_dim,2)
        (lower bound, upper bound) for every dimension
        If a value is set to `float('inf')`, the bound will be assumed to be the min/max value of the dimension
        Also, if `limits == None`, the limits are defined by the maximum and minimum value of the lens for all dimensions.
        i.e. `[[min_1, max_1], [min_2, max_2], [min_3, max_3]]`

    Example
    ---------

    ::

        >>> import numpy as np
        >>> from kmapper.cover import Cover
        >>> data = np.random.random((100,2))
        >>> cov = Cover(n_cubes=15, perc_overlap=0.75)
        >>> cube_centers = cov.fit(data)
        >>> cov.transform_single(data, cube_centers[0])
        array([[0.3594448 , 0.07428465],
               [0.14490332, 0.01395559],
               [0.94988668, 0.03983579],
               [0.73517978, 0.09420806],
               [0.16903735, 0.06901085],
               [0.81578595, 0.10708731],
               [0.26923572, 0.12216203],
               [0.89203167, 0.0711279 ],
               [0.80442115, 0.10220901],
               [0.33210782, 0.04365007],
               [0.52207707, 0.05892861],
               [0.26589744, 0.08502856],
               [0.02360067, 0.1263653 ],
               [0.29855631, 0.01209373]])
        >>> hyper_cubes = cov.transform(data, cube_centers)

    """

    def __init__(self, n_cubes=10, perc_overlap=0.5, limits=None, verbose=0):
        self.centers_ = None
        self.radius_ = None
        self.inset_ = None
        self.inner_range_ = None
        self.bounds_ = None
        self.di_ = None

        self.n_cubes = n_cubes
        self.perc_overlap = perc_overlap
        self.limits = limits
        self.verbose = verbose

        # Check limits can actually be handled and are set appropriately
        assert isinstance(
            self.limits, (list, np.ndarray, type(None))
        ), "limits should either be an array or None"
        if isinstance(self.limits, (list, np.ndarray)):
            self.limits = np.array(self.limits)
            assert self.limits.shape[1] == 2, "limits should be (n_dim,2) in shape"

    def __repr__(self):
        return "Cover(n_cubes=%s, perc_overlap=%s, limits=%s, verbose=%s)" % (
            self.n_cubes,
            self.perc_overlap,
            self.limits,
            self.verbose,
        )

    def _compute_bounds(self, data):

        # If self.limits is array-like
        if isinstance(self.limits, np.ndarray):
            # limits_array is used so we can change the values of self.limits from None to the min/max
            limits_array = np.zeros(self.limits.shape)
            limits_array[:, 0] = np.min(data, axis=0)
            limits_array[:, 1] = np.max(data, axis=0)
            limits_array[self.limits != float("inf")] = 0
            self.limits[self.limits == float("inf")] = 0
            bounds_arr = self.limits + limits_array
            """ bounds_arr[i,j] = self.limits[i,j] if self.limits[i,j] == inf
                bounds_arr[i,j] = max/min(data[i]) if self.limits == inf """
            bounds = (bounds_arr[:, 0], bounds_arr[:, 1])

            # Check new bounds are actually sensible - do they cover the range of values in the dataset?
            if not (
                (np.min(data, axis=0) >= bounds_arr[:, 0]).all()
                or (np.max(data, axis=0) <= bounds_arr[:, 1]).all()
            ):
                warnings.warn(
                    "The limits given do not cover the entire range of the lens functions\n"
                    + "Actual Minima: %s\tInput Minima: %s\n"
                    % (np.min(data, axis=0), bounds_arr[:, 0])
                    + "Actual Maxima: %s\tInput Maxima: %s\n"
                    % (np.max(data, axis=0), bounds_arr[:, 1])
                )

        else:  # It must be None, as we checked to see if it is array-like or None in __init__
            bounds = (np.min(data, axis=0), np.max(data, axis=0))

        return bounds

    def fit(self, data):
        """Fit a cover on the data. This method constructs centers and radii in each dimension given the `perc_overlap` and `n_cube`.

        Parameters
        ============

        data: array-like
            Data to apply the cover to. Warning: First column must be an index column.

        Returns
        ========

        centers: list of arrays
            A list of centers for each cube

        """

        # TODO: support indexing into any columns
        di = np.array(range(1, data.shape[1]))
        indexless_data = data[:, di]
        n_dims = indexless_data.shape[1]

        # support different values along each dimension

        ## -- is a list, needs to be array
        ## -- is a singleton, needs repeating
        if isinstance(self.n_cubes, Iterable):
            n_cubes = np.array(self.n_cubes)
            assert (
                len(n_cubes) == n_dims
            ), "Custom cubes in each dimension must match number of dimensions"
        else:
            n_cubes = np.repeat(self.n_cubes, n_dims)

        if isinstance(self.perc_overlap, Iterable):
            perc_overlap = np.array(self.perc_overlap)
            assert (
                len(perc_overlap) == n_dims
            ), "Custom cubes in each dimension must match number of dimensions"
        else:
            perc_overlap = np.repeat(self.perc_overlap, n_dims)

        assert all(0.0 <= p <= 1.0 for p in perc_overlap), (
            "Each overlap percentage must be between 0.0 and 1.0., not %s"
            % perc_overlap
        )

        bounds = self._compute_bounds(indexless_data)
        ranges = bounds[1] - bounds[0]

        # (n-1)/n |range|
        inner_range = ((n_cubes - 1) / n_cubes) * ranges
        inset = (ranges - inner_range) / 2

        # |range| / (2n ( 1 - p))
        with np.errstate(divide='ignore'):
            radius = ranges / (2 * (n_cubes) * (1 - perc_overlap))

        # centers are fixed w.r.t perc_overlap
        zip_items = list(bounds)  # work around 2.7,3.4 weird behavior
        zip_items.extend([n_cubes, inset])
        centers_per_dimension = [
            np.linspace(b + r, c - r, num=n) for b, c, n, r in zip(*zip_items)
        ]
        centers = [np.array(c) for c in product(*centers_per_dimension)]

        self.centers_ = centers
        self.radius_ = radius
        self.inset_ = inset
        self.inner_range_ = inner_range
        self.bounds_ = bounds
        self.di_ = di

        if self.verbose > 0:
            print(
                " - Cover - centers: %s\ninner_range: %s\nradius: %s"
                % (self.centers_, self.inner_range_, self.radius_)
            )

        return centers

    def transform_single(self, data, center, i=0):
        """Compute entries of `data` in hypercube centered at `center`

        Parameters
        ===========

        data: array-like
            Data to find in entries in cube. Warning: first column must be index column.
        center: array-like
            Center points for the cube. Cube is found as all data in `[center-self.radius_, center+self.radius_]`
        i: int, default 0
            Optional counter to aid in verbose debugging.
        """

        lowerbounds, upperbounds = center - self.radius_, center + self.radius_

        # Slice the hypercube
        entries = (data[:, self.di_] >= lowerbounds) & (
            data[:, self.di_] <= upperbounds
        )
        hypercube = data[np.invert(np.any(entries == False, axis=1))]

        if self.verbose > 1:
            print(
                "There are %s points in cube %s/%s"
                % (hypercube.shape[0], i + 1, len(self.centers_))
            )

        return hypercube

    def transform(self, data, centers=None):
        """Find entries of all hypercubes. If `centers=None`, then use `self.centers_` as computed in `self.fit`.

            Empty hypercubes are removed from the result

        Parameters
        ===========

        data: array-like
            Data to find in entries in cube. Warning: first column must be index column.
        centers: list of array-like
            Center points for all cubes as returned by `self.fit`. Default is to use `self.centers_`.

        Returns
        =========
        hypercubes: list of array-like
            list of entries in each hypercube in `data`.

        """

        centers = centers or self.centers_
        hypercubes = [
            self.transform_single(data, cube, i) for i, cube in enumerate(centers)
        ]

        # Clean out any empty cubes (common in high dimensions)
        hypercubes = [cube for cube in hypercubes if len(cube)]
        return hypercubes

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def find(self, data_point):
        """Finds the hypercubes that contain the given data point.

        Parameters
        ===========

        data_point: array-like
            The data point to locate.

        Returns
        =========
        cube_ids: list of int
            list of hypercube indices, empty if the data point is outside the cover.

        """
        cube_ids = []
        for i, center in enumerate(self.centers_):
            lower_bounds, upper_bounds = center - self.radius_, center + self.radius_
            if np.all(data_point >= lower_bounds) and np.all(
                data_point <= upper_bounds
            ):
                cube_ids.append(i)
        return cube_ids


class CubicalCover(Cover):
    """
    Explicit definition of a cubical cover as the default behavior of the cover class. This is currently identical to the default cover class.
    """

    pass
