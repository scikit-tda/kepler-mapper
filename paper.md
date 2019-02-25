---
title: 'Kepler Mapper: A Lean Persistent Homology Library for Python'
tags:
 - Python
 - Mapper
 - Topological Data Analysis
authors:
 - name: Hendrik Jacob van Veen
   orcid: XXXX-XXXX-XXXX-XXX
   affiliation: 1
 - name: Nathaniel Saul
   orcid: 0000-0002-8549-9810
   affiliation: 2
 - name: Emilie (empet)
 - name: deargle
 - name: smangham
 - name: emerson-escolar
affiliations:
 - name: Nubank
   index: 1
 - name: Department of Mathematics and Statistics, Washington State University
   index: 2
date: 12 February 2018
bibliography: paper.bib
---

# Summary

Topological data analysis (TDA) is a field analysis focused on understanding the shape and structure of data.  By computing topological descriptors of data such as connected components, loops, and voids, we are better able to find hidden relationships [@edelsbrunner2010computational], [@carlsson2009topology]. Mapper is one main technique from the field that is designed for visualization of high dimensional topological structures.

This work provides an easy to use implementation for the Mapper algorithm. We leverage Scikit-Learn API compatible cluster and scaling algorithms to construct Mappers in a flexible and user friendly way. 

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as first described in the paper “Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition” [@Singh2007]. 
It has since become the backbone technology of the enterprise AI company Ayasdi. 
The work of [@Lumetal2013] sparked widespread interest in the technique by demonstrating the use in multiple domains, such as political science, biology, and sports analytics. 


In this work, we develop an intuitive interface for Mapper algorithm along with multiple comprehension methods for visualizing the construction. We also an provide extensive suite of tutorials detailing the use of Mapper for simple and complex use cases.


# Library Details

Kepler Mapper provides an object oriented API for constructing lenses and building Mapper. The module employs the strategy pattern [cite book? or wikipedia] so uses can specify the clustering algorithm, covering scheme, and nerve scheme.  This allows the module to be flexible for many use cases. In the examples and often in practice, clustering strategies abide by the scikit-learn clusterer interface [@scikit-learn]. We have developed similar interfaces for `Cover` classes and `Nerve` classes and provided default implementations that are most commonly found in the literature. 

Visual exploration is a critical portion of Mapper analysis. For this, we provide multiple methods for visualization. For interactive visualization and exploration in the browser, the Mapper class can output as an HTML file utilizing D3.js. For use with Jupyter or embedded purposes, we provide a visualization interface utilizing Plotly. For static visualizations, we provide an adapter for networkx and visualization using matplotlib.


# Source Code

The source code for Kepler Mapper is available on Github through the Scikit-TDA organization [https://github.com/scikit-tda/kepler-mapper](https://github.com/scikit-tda/kepler-mapper).   

# Acknowledgements

Nathaniel Saul was partially supported by NSF DBI-1661348 and by Washington NASA Space Grant Consortium, NASA Grant #NNX15AJ98H. 

# References
