# Release log

## Unreleased

### Added

- ability to live-update the min-intersection threshold for edges on the d3 vis (#231)

### Fixed/Changed

- Update docs and sphinx-gallery to build again.
- Update CI/CD runners to test modern versions of python; drop python 3.7 compatibility.
- Change visual tests to use built-in `int` type rather than `np.uint8`.

## 2.0.1

### Fixed

- `min_cluster_samples` now only accepts an int -- now AgglomerativeClustering works (#224)
- `plotlyviz.scomplex_to_graph` no longer casts `color_values` to a 2d array, and `visuals._tooltip_components` now generates
  either 1d or 2d `member_histogram` depending on dimensionality of `color_values` (#225)

### Fixed/Changed

- The AND and OR searchbar queries no longer multiplies the base size of a node by the ratio of how many of its items match. Rather,
  the base size of the node is simply multiplied by how many of its items match the query. With this change, the size of a node
  during an AND or OR search query again directly reflects the number of items within the node. (#227)
- The default search mode is now AND -- because that's the expected behavior, because that's how the google works (#227)

## 2.0.0

### Visualization

All of the below changes apply to kmapper's D3 html visualization.

- added searchbar functionality (see documentation for `include_searchbar` in `kmapper.visualize()`)
- upgraded d3 from v3 to v6 -- one huge benefit of the new d3-force library is that it is [deterministic](https://
  twitter.com/mbostock/status/725124754701717504?lang=en), so a given graph will always render the same visually, across browsers and reloads.
- clicking on a node will set it as the "focus node" (the node for which details are shown in the details pane).
  until the user clicks off of the node. That is to say, click-focus is not lost if another node is moused-over.
  Click-focus is released if (1) another node is clicked on, or (2) if the user clicks on the blank canvas.
- hovering over a node will "freeze" it in place until no longer hovering over that node. This makes it easier to
  grab the node.
  If no node is currently set as the "focus node" via a click, then hovering over a node will also make it the focus node.
- once a node is dragged, it stays ("freezes") where it was dragged
- added the ability to freeze (and unfreeze) all nodes with keystrokes f and x,
- the focus node visually "pulses" in the display
- added the ability to "save" the positioning of all nodes in the display. Saves to a .json file.
  Node positioning can be re-loaded via providing the json save file.
- multiple `color_values` arrays can be passed, and switched between interactively in the display.
- the node color function can be specified, as a string, to any function available on the numpy base class (e.g.,
  'mean', 'median', 'max', 'min'. (Before, the only available function was `np.mean`.
  - Multiple node color functions can be specified, and toggled between interactively in the display.
- The toolbar display now uses css flexbox, which avoids overlap-problems on smaller viewports.

### Kmapper

- change several visualize-related functions to be private
- only support python >= 3.6

## 1.4.1

- New CI/CD pipeline

## 1.4

- More flexible visualization coloring (PR 190)
- Better support for sparse matrices (PR 189)
- Better support for precomputed distance matrices (PR 184)

## 1.3.x

- A series of releases to support JOSS submission

## 1.3.0 (October 12, 2019)

- JOSS Release -- final revision
- Allow sparse matrices in `map` function (PR #163)
- Use sphinx-gallery for documentation examples (#164)
- Removed mutable arguments (#165)

## 1.2.0 (Feb 18, 2019)

- New implementation of the cover API makes it consistent with the literature (your % overlap will probably have to be decreased when updating).
- New documentation website (kepler-mapper.scikit-tda.org).

## 1.1.6 (Nov 8, 2018)

- Plotly visualization interface.
- Networkx adapter
- Bug fixes
- Scikit-tda integration

## 1.1.2

- Bug fix, setup.py did not include static directory so installation visualizations did not work when installed from pypi.
- Add Jupyter notebook support

## 1.1

- Massive visualization upgrades
- Separation of HTML, JS, CSS, and Python code
- New nerves and covers API
- Documentation site

## 1.0.1

- Convert versioning scheme to major.minor.micro
- Restructure library to be compatible with PyPi installation
- Minor bug fixes
- Include preliminary unit test suite
- Refactor, extract helper classes and helper functions

# Release log - Pre-alpha

## v00009

## v00008

## v00007

- Add L2^Norm Lens
- Add Winsconsin Breast Cancer Data Anomaly Detection Example
- Fixed bug: k-means with set number of clusters higher than min_cluster_samples
- Add self.inverse_X for new future feature: transforming on unseen data.

## v00006

- Removed link_local functionality
- Halved the number of edges drawn (no two-way edges)
- Added support for clustering on the inverse image
- Refactored code (see updated documentation)
- Added code comments
- Added feature to use reducers/manifold learning/dimensions and stat functions
- Added 7 projections/lenses from statistics

## v00005

- Made Python 3 compatible
- Ability to turn off title, meta and tooltips
- Ability to set the window height and width of HTML output
- Added basic support for another color function: average signal
- De-emphasized link_local functionality, since its current implementation is no good.

## v00004

- Added dimensionality reduction
- Added "digits" case study
- changed fit to fit_transform and return of data
- added tooltips
- added support for custom tooltips

## v00003

- Refactored dimension index to use a list of arbitrary dimensions
- Improved verbosity
- Added levels of verbosity
- Decreased number of code lines by using a single approach
- Added sample to explain local linkage True vs. False
- Added side-view for animal point-cloud data
- Added a gallery in the example directory

## v00002

- Added a multi-dimensional mode: use all dimensions.
- Added case study: 3D point cloud data for animals
- Added case study: Make circles
- Added advanced parameters for graph layout settings. Should probably be sliders on the .html page itself.
- Improved documentation
- Added disclaimer
- Added todo
- Added release log

## v00001

- Wrote class
- Wrote documentation
- Added license
