import numpy as np
from sklearn.datasets import make_circles
from kmapper import KeplerMapper

from kmapper.visuals import init_color_function, format_meta, format_mapper_data



np.random.seed(1)


"""
    Interested in rebuilding the API of kepler mapper so it is more intuitive


    Should Kepler Mapper be split into two objects?

    I don't get how distance_matrix works

    The visualize method should have sane defaults.
        
    Tooltips   
        - Tooltips should default to showing the ID of data point in each node.
        - Tooltips should be able to be disabled.
        - Tooltips should be able to show aggregate data for each node.
        - Tooltips should easily be able to export the data.

    Graph
        - Graph should be able to be frozen.
        - Graph should be able to switch between multiple coloring functions.
        - Should be able to remove nodes from graph (so you can clean out noise)
        - Edits should be able to be saved. Can re-export the html file so you can open it in the same state.
        - Color funcs should be easier to use.
        - Should be able to choose any D3 palette
        - Cold is low, hot is high.
    
    Style:
        - 'inverse_X' should just be called X
        - More of the html stuff should be in the jinja2 stuff.
        - If running from source, should be able to run offline

    Map 
        - Move all of these arguments into the init method

"""



class TestVisualHelpers():
    def test_color_function_type(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        graph = {"nodes": nodes}

        color_function = init_color_function(graph)

        assert type(color_function) == np.ndarray
        assert min(color_function) == 0
        assert max(color_function) == 1

    def test_color_function_scaled(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6]}
        graph = {"nodes": nodes}

        color_function = init_color_function(graph)

        assert min(color_function) == 0
        assert max(color_function) == 1

    def test_color_function_size(self):
        nodes = {"a": [1, 2, 3], "b": [4, 5, 6, 7, 8, 9]}
        graph = {"nodes": nodes}

        color_function = init_color_function(graph)

        assert len(color_function) == len(nodes['a']) + len(nodes['b']) + 1

    def test_format_meta(self):
        mapper = KeplerMapper()
        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        assert("<p>%s</p>"%(len(graph["nodes"])) in format_meta(graph))
        assert("<h3>Description</h3>\n<p>A short description</p>" in 
            format_meta(graph,
                custom_meta=[("Description", "A short description")]))

    def test_format_mapper_data(self):
        mapper = KeplerMapper()
        data, labels = make_circles(1000, random_state=0)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)

        color_function = lens[:,0]
        inverse_X = data
        projected_X = lens
        projected_X_names = ["projected_%s"%(i) for i in range(projected_X.shape[1])]
        inverse_X_names = ["inverse_%s"%(i) for i in range(inverse_X.shape[1])]
        custom_tooltips = np.array(["customized_%s"%(l) for l in labels])

        graph_data = format_mapper_data(graph, color_function, inverse_X,
                 inverse_X_names, projected_X, projected_X_names, custom_tooltips)

        # TODO test more properties!
        assert 'name' in graph_data['nodes'][0].keys()
        # assert "cube2_cluster0" == graph_data['name']
        # assert """projected_0""" in graph_data.keys()
        # assert """inverse_0""" in graph_data
        # assert """customized_""" in graph_data


class TestVisualizeIntegration():
    def test_visualize_standalone_same(self, tmpdir):
        """ ensure that the visualization is not dependent on the actual mapper object.
        """
        mapper = KeplerMapper()

        file = tmpdir.join('output.html')

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz1 = mapper.visualize(graph, path_html=file.strpath)

        new_mapper = KeplerMapper()
        viz2 = new_mapper.visualize(graph, path_html=file.strpath)

        assert viz1 == viz2

    def test_file_written(self, tmpdir):
        mapper = KeplerMapper()

        file = tmpdir.join('output.html')

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz = mapper.visualize(graph, path_html=file.strpath)

        assert file.read() == viz
        assert len(tmpdir.listdir()) == 1, "file was written to"

    def test_file_not_written(self, tmpdir):
        mapper = KeplerMapper()

        file = tmpdir.join('output.html')

        data = np.random.rand(1000, 10)
        lens = mapper.fit_transform(data, projection=[0])
        graph = mapper.map(lens, data)
        viz = mapper.visualize(graph, path_html=file.strpath, save_file=False)

        assert len(tmpdir.listdir()) == 0, "file was never written to"
        # assert file.read() != viz

