#!/usr/bin/env python

from distutils.core import setup

setup(name='kmapper',
      version='1.1.2',
      description='Python implementation of Mapper algorithm for Topological Data Analysis.',
      long_description="""
This is a Python implementation of the TDA Mapper algorithm for visualization of high-dimensional data. For complete documentation, see https://MLWave.github.io/kepler-mapper.

KeplerMapper employs approaches based on the Mapper algorithm (Singh et al.) as first described in the paper "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition".

KeplerMapper can make use of Scikit-Learn API compatible cluster and scaling algorithms.
""",
      author='HJ van Veen, Nathaniel Saul',
      author_email='info@mlwave.com, nat@saulgill.com',
      url='https://MLWave.github.io/kepler-mapper',
      license='MIT',
      packages=['kmapper'],
      install_requires=[
        'scikit-learn',
        'numpy',
        'scipy',
        'Jinja2'
      ],
      test_requires=[
        'pytest'
      ],
      python_requires='>=2.7,!=3.1,!=3.2,!=3.3',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='mapper, topology data analysis, algebraic topology, unsupervised learning',
      include_package_data=True
     )
