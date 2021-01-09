# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from kmapper import __version__
from sktda_docs_config import *

project = u'KeplerMapper'
copyright = u'2019, Hendrik Jacob van Veen and Nathaniel Saul'
author = u'Hendrik Jacob van Veen and Nathaniel Saul'

version = __version__
release = __version__

html_static_path = ["../examples/output"]

# extensions.append('sphinx_gallery.gen_gallery')

import pathlib
path = pathlib.Path.cwd()
example_dir = path.parent.joinpath('examples')
# sphinx_gallery_conf = {
#      'examples_dirs': example_dir,   # path to your example scripts
#      'gallery_dirs': path.joinpath('generated', 'gallery'),  # path where to save gallery generated examples
#      'image_scrapers': ('matplotlib',),
#      'abort_on_example_error': True,
#      'plot_gallery': True
# }

html_theme_options.update({
  # Google Analytics info
  'ga_ua': 'UA-124965309-4',
  'ga_domain': '',
  'gh_url': 'scikit-tda/kepler-mapper'
})

html_short_title = project
htmlhelp_basename = 'KeplerMapperdoc'