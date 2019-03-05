Background
-------------

Topology started with Leonhard Euler and the famous problem of the Seven Bridges of Königsberg: Can one construct a path that crosses each bridge exactly once and reaches all islands?

The fundamental assumption in topology is that connectivity is more important than distance. This lack of distinction is important to why topology is useful for analyzing data. Instead of getting wrapped up in myopic details of how far apart two points are, topology is concerned with qualitative questions, like: how many holes does the object have?, or: how many pieces is it constructed out of? Essentially, topology is a way to explore the shape of data without concern for things like which metric to use.

Fundamental shapes of data that can be studied with topology:

- linearities
- non-linearities
- clusters
- flares
- loops

Real-world data is often complex and contains multiple different fundamental shapes.

Topological Data Analysis
===========================


In applied mathematics, topological data analysis (TDA) is an approach to the analysis of datasets using techniques from topology. Extraction of information from datasets that are high-dimensional, incomplete and noisy is generally challenging. TDA provides a general framework to analyze such data in a manner that is insensitive to the particular metric chosen and provides dimensionality reduction and robustness to noise. Beyond this, it inherits functoriality, a fundamental concept of modern mathematics, from its topological nature, which allows it to adapt to new mathematical tools.

TDA offers:

- **A lossy compressed mathematical representation of a data set.** You can study the global structure of a dataset, down to the details of a single data point, without incurring a cognitive overload.
- **Resistance to noise and missing data.** TDA retains significant features of the data.
- **Invariance.** Only connectedness matters. The skew, size, or orientation of data does not fundamentally change that data.
- **A data exploration tool.** Get answers to questions you haven't even asked yet.
- **A methodology to study the shape of data and manifolds.** TDA has a solid theoretical foundation and inherits functoriality.

Mapper
=========

Mapper is way to construct a graph (or simplicial complex) from data in a way that reveals the some of the topological features of the space. Though not exact, mapper is able to estimate important connectivity aspects of the underlying space of the data so that it can be explored in a visual format. This is an unsupervised method of generating a visual representation of the data that can often reveal new insights of the data that other methods cannot.

Informally, the Mapper algorithm works by performing a local clustering guided by a projection function. The steps are as follows:

- **Project** a dataset (for instance: :code:`mean of x for x in X`, or :code:`t-SNE(2d).fit_transform(X))`. You can use any projection function from maths, statistics, econometrics, or machine learning.
- **Cover** this projection with overlapping intervals/hypercubes. In theory you can use any interval shape, but KeplerMapper currently supports n-dimensional hypercubes.
- **Cluster** the points inside an interval (either apply clustering on the projection and suffer projection loss, or cluster on the inverse image/original data). You can use any clustering algorithm (hierarchical, density-based, etc.) and distance metric (does not need to be a proper metric that satisfies triangle inequality).
- The **clusters become nodes in a graph.**
- Due to the overlap, a single point can appear in multiple nodes. When there is such a member intersection, **draw an edge** between these nodes.
- Now to aid visual exploration:
    - Size the graph nodes by a function of interest (for instance: number of members inside cluster)
    - Color the graph nodes by a function of interest (for instance: average customer spend)
    - Shape the graph nodes by a function of interest (for instance: squares for :code:`avg_timestamp < 2015`, circles for :code:`avg_timestamp >= 2015`)
    - Size the graph edges by a function of interest (for instance: number of intersecting members between nodes)
    - Color the graph edges by a function of interest (for instance: average color of connected nodes)
    - Shape the graph edges by a function of interest (for instance: dotted line when intersecting :code:`members < 3` else solid line)
    - **Provide descriptive statistics** on the nodes and the graph. (for instance: Kolmogorov–Smirnov test between node variables and dataset variables.





