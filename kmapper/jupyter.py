import IPython
import html as HTML

# Here we set the custom CSS to override Jupyter's default
CUSTOM_CSS = """<style>
    .container { width:100% !important; }
    .output_scroll {height: 800px !important;}
    </style>"""
IPython.core.display.display(IPython.core.display.HTML(CUSTOM_CSS))

def display(path_html="mapper_visualization_output.html",
            html=None,
            width=100,
            height=800):
    """ Displays a html file inside a Jupyter Notebook output cell.

    Parameters
    ----------
    path_html : str
        Path to html. Use file name for file inside current working
        directory. Use `file://` browser url-format for path to local file.
        Use `https://` urls for externally hosted resources.

    Notes
    -----
    Thanks to https://github.com/smartinsightsfromdata for the issue:
    https://github.com/MLWave/kepler-mapper/issues/10

    """


    if html:
        html = HTML.escape(html)
        iframe = f'''<iframe srcdoc="{html}" width={width}%% height={height}></iframe>'''
        IPython.core.display.display(IPython.display.HTML(iframe))

    else:
        iframe = '<iframe src=' + path_html \
                + f' width={width}%% height={height} frameBorder="0"></iframe>'

        IPython.core.display.display(IPython.core.display.HTML(iframe))
