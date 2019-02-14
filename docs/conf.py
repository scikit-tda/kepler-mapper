# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from kmapper import __version__
from theme_settings import *

project = u'KeplerMapper'
copyright = u'2019, Hendrik Jacob van Veen and Nathaniel Saul'
author = u'Hendrik Jacob van Veen and Nathaniel Saul'

version = __version__
release = __version__

html_theme_options.update({
  # Google Analytics info
  'ga_ua': 'UA-124965309-4',
  'ga_domain': '',
})

html_short_title = project
htmlhelp_basename = 'KeplerMapperdoc'