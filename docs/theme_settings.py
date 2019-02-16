from better import better_theme_path

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',
    "sphinx.ext.napoleon",
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinxcontrib.fulltoc'
]

master_doc = 'index'


autodoc_default_options = {
    'autoclass_content': "both"
}
autodoc_default_flags = [
    "members",
    "inherited-members"
]
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

html_logo = "logo.png"
source_suffix = '.rst'



html_theme_path = [better_theme_path]
html_theme = 'better'
html_sidebars = {
    '**': [
        'localtoc.html', 
        'sourcelink.html', 
        'searchbox.html'
    ],
}

html_static_path = ['_static']
templates_path = ['_templates']
html_theme_options = {
#   'cssfiles': 'custom_style.css',
 
  
  # show sidebar on the right instead of on the left
  'rightsidebar': False,

  # inline CSS to insert into the page if you're too lazy to make a
  # separate file
#   'inlinecss': '',

  # CSS files to include after all other CSS files
  # (refer to by relative path from conf.py directory, or link to a
  # remote file)
  'cssfiles': ['_static/custom_style.css'],  # default is empty list

  # show a big text header with the value of html_title
  'showheader': True,

  # show the breadcrumbs and index|next|previous links at the top of
  # the page
  'showrelbartop': True,
  # same for bottom of the page
  'showrelbarbottom': True,



  # show the self-serving link in the footer
  'linktotheme': True,

  # width of the sidebar. page width is determined by a CSS rule.
  # I prefer to define things in rem because it scales with the
  # global font size rather than pixels or the local font size.
  'sidebarwidth': '15rem',

  # color of all body text
  'textcolor': '#000000',

  # color of all headings (<h1> tags); defaults to the value of
  # textcolor, which is why it's defined here at all.
  'headtextcolor': '',

  # color of text in the footer, including links; defaults to the
  # value of textcolor
  'footertextcolor': '',


}

nbsphinx_allow_errors = True
language = None
exclude_patterns = [u'_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = 'sphinx'