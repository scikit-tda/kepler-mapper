import IPython



def display(path_html="mapper_visualization_output.html"):
    """ Displays a html file inside a Jupyter Notebook output cell.
        
        .. note::
        
            Must run ``KeplerMapper.visualize`` first to generate html. This function will then render that output from a file saved to disk.
    
        .. note::

            Thanks to `smartinsightsfromdata <https://github.com/smartinsightsfromdata>`_ for the `github issue 10 <https://github.com/MLWave/kepler-mapper/issues/10>`_ that suggested this method. 

    Parameters
    ============

    path_html : str
        Path to html. Use file name for file inside current working 
        directory. Use ``file://`` browser url-format for path to local file.
        Use ``https://`` urls for externally hosted resources.

        


    Examples
    =========

    ::

        import numpy as np
        import kmapper as km
        from kmapper.jupyter import display

        data = np.random.random((2000, 2))
        mapper = km.KeplerMapper()
        lens = km.project(data)
        graph = km.map(lens, data)
        _ = km.visualize(graph, path_html="filename.html")

        display("filename.html")


    The default filename is the same default as the ``.visualize`` method, so using both without arguments will show the last constructed graph:

    >>> _ = km.visualize(graph)
    >>> display()

    """

    # Here we set the custom CSS to override Jupyter's default
    CUSTOM_CSS = """<style>
        .container { width:100% !important; }
        .output_scroll {height: 800px !important;}
        </style>"""
    IPython.core.display.display(IPython.core.display.HTML(CUSTOM_CSS))

    iframe = (
        "<iframe src=" + path_html + ' width=100%% height=800 frameBorder="0"></iframe>'
    )
    IPython.core.display.display(IPython.core.display.HTML(iframe))
