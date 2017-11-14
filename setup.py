#!/usr/bin/env python

from distutils.core import setup

setup(name='kmapper',
      version='0.2.1',
      description='Python implementation of mapper algorithm for Topological Data Analysis.',
      author='HJ van Veen',
      author_email='info@mlwave.com',
      url='https://github.com/MLWAve/kepler-mapper',
      packages=['kmapper'],
      install_requires=[
        'scikit-learn',
        'numpy',
        'scipy'
      ],
      test_requires=[
        'pytest'
      ]
     )
