# -*- coding: utf-8 -*-
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from kmapper import __version__
from sktda_docs_config import *

project = "KeplerMapper"
copyright = (
    "2019-%s, Hendrik Jacob van Veen, Nathaniel Saul, David Eargle, and Sam Mangham"
    % time.strftime("%Y")
)
author = "Hendrik Jacob van Veen, Nathaniel Saul, David Eargle, and Sam Mangham"
language = "en"

version = __version__
release = __version__

html_static_path = ["../examples/output", "_static", "notebooks/output"]

templates_path = ["_templates"]
exclude_patterns.append("_templates")
# exclude_patterns.append('generated')
# exclude_patterns.append('generated/gallery')

extensions.append("sphinx_gallery.gen_gallery")

import pathlib

path = pathlib.Path.cwd()
example_dir = path.parent.joinpath("examples")
sphinx_gallery_conf = {
    "examples_dirs": example_dir,  # path to your example scripts
    "gallery_dirs": path.joinpath(
        "generated", "gallery"
    ),  # path where to save gallery generated examples
    "image_scrapers": ("matplotlib",),
    "abort_on_example_error": True,
    "copyfile_regex": r".*\.csv",
    # 'run_stale_examples': True,
    # 'plot_gallery': True,
    "remove_config_comments": True,
    "thumbnail_size": (400, 280),
}

html_theme_options.update(
    {
        # Google Analytics info
        "ga_ua": "UA-124965309-4",
        "ga_domain": "",
        "gh_url": "scikit-tda/kepler-mapper",
        "cssfiles": ["_static/gallery-override.css"],
    }
)

# sphinx 4 defaults to mathjax3, but that is not working with the theme right now.
# manually set mathjax2.
mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

nbsphinx_allow_errors = False

# uncomment to rebuild all jupyter notebooks (for testing)
# Warning, some of the notebooks take a long time to run!
#
# nbsphinx_execute = 'always'


def setup(app):
    app.add_css_file("gallery-override.css")


html_short_title = project
htmlhelp_basename = "KeplerMapperdoc"

autodoc_default_options = {"members": True}

autodoc_member_order = "groupwise"
