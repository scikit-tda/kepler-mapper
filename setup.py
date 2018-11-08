#!/usr/bin/env python

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='kmapper',
      version='1.1.6',
      description='Python implementation of Mapper algorithm for Topological Data Analysis.',
      long_description=long_description,
      long_description_content_type="text/markdown",	
      author='HJ van Veen, Nathaniel Saul',
      author_email='info@mlwave.com, nat@saulgill.com',
      url='http://kepler-mapper.scikit-tda.org',
      license='MIT',
      packages=['kmapper'],
      include_package_data=True,
      extras_require={
        'testing': [
          'pytest',
          'networkx',
          'matplotlib',
          'python-igraph',
          'plotly',
          'ipywidgets'   
        ]
      },
      install_requires=[
        'scikit-learn',
        'numpy',
        'scipy',
        'Jinja2'
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
      keywords='mapper, topology data analysis, algebraic topology, unsupervised learning'
     )
