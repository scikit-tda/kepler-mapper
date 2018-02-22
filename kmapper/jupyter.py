import IPython

CUSTOM_CSS = """<style>
    .container { width:100% !important; }
    .output_scroll {height: 800px !important;}
    </style>"""
IPython.core.display.display(IPython.core.display.HTML(CUSTOM_CSS))

def display(path_html="mapper_visualization_output.html"):
    iframe = '<iframe src=' + path_html \
            + ' width=100%% height=800 frameBorder="0"></iframe>'
    IPython.core.display.display(IPython.core.display.HTML(iframe))