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

Topological data analysis (TDA) is a field analysis focused on understanding the shape and structure of data.  By computing topological descriptors of data such as connected components, loops, and voids, we are better able to find hidden relationships [@edelsbrunner2010computational, @carlsson2009topology]. Mapper is one main technique from the field that is designed for visualization of high dimensional topological structures.

This work provides an easy to use implementation for the Mapper algorithm. We leverage Scikit-Learn API compatible cluster and scaling algorithms to construct Mappers in a flexible and user friendly way. 

KeplerMapper employs approaches based on the MAPPER algorithm (Singh et al.) as first described in the paper “Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition” [@SiMeCa2007]. 
It has since become the backbone technology of the enterprise AI company Ayasdi. 
The work of [@Lumetal2013] sparked widespread interest in the technique by demonstrating the use in multiple domains, such as political science, biology, and sports analytics. 


In this work, we develop an intuitive interface for Mapper algorithm along with multiple comprehension methods for visualizing the construction. We also an provide extensive suite of tutorials detailing the use of Mapper for simple and complex use cases.


# Library Details

Ripser.py supplies two interfaces: one lightweight and functional interface, as well as an object-oriented interface designed to fit within the Scikit-Learn transformer paradigm [@scikit-learn]. We have merged together multiple branches of the original Ripser library ("sparse-distance-matrix," "representative-cocycles") to expose some lesser known but incredibly useful features hidden in Ripser.  Below we detail some of the special features made easy with our library.



# Source Code
The source code for Kepler Mapper is available on Github through the Scikit-TDA organization [https://github.com/scikit-tda/kepler-mapper](https://github.com/scikit-tda/kepler-mapper).   

# Acknowledgements

Nathaniel Saul was partially supported by NSF DBI-1661348 and by Washington NASA Space Grant Consortium, NASA Grant #NNX15AJ98H. 

# References
