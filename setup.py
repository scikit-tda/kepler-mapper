#!/usr/bin/env python

from distutils.core import setup

setup(name='kmapper',
      version='1.0.1',
      description='Python implementation of mapper algorithm for Topological Data Analysis.',
      long_description='Python implementation of mapper algorithm. This implementation is intended to integrate into scikit-learn and be easily extendible.',
      author='HJ van Veen, Nathaniel Saul',
      author_email='info@mlwave.com, nat@saulgill.com',
      url='https://github.com/MLWAve/kepler-mapper',
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
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',

        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='mapper, topology data analysis, algebraic topology, unsupervised learning'


     )
