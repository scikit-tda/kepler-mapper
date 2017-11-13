#!/usr/bin/env python

from distutils.core import setup

setup(name='kmapper',
      version='0.2.1',
      description='Kepler-mapper',
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
