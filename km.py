from __future__ import division
import numpy as np
from collections import defaultdict
import json
import itertools
from sklearn import cluster, preprocessing, manifold
from datetime import datetime
import sys

class KeplerMapper(object):
  def __init__(self, cluster_algorithm=cluster.DBSCAN(eps=0.5,min_samples=3), nr_cubes=10, 
         overlap_perc=0.1, scaler=preprocessing.MinMaxScaler(), reducer=None, color_function="distance_origin", 
         link_local=False, verbose=1):
    self.clf = cluster_algorithm
    self.nr_cubes = nr_cubes
    self.overlap_perc = overlap_perc
    self.scaler = scaler
    self.color_function = color_function
    self.verbose = verbose
    self.link_local = link_local
    self.reducer = reducer
    
    self.chunk_dist = []
    self.overlap_dist = []
    self.d = []
    
    if self.verbose > 0:
      print("\nnr_cubes = %s \n\noverlap_perc = %s\n\nlink_local = %s\n\nClusterer = %s\n\nScaler = %s\n\n"%(self.nr_cubes, overlap_perc, self.link_local, str(self.clf),str(self.scaler)))
  
  def fit_transform(self, X):
    # Dimensionality Reduction
    if self.reducer != None:
      if self.verbose > 0:
        try:    
          self.reducer.set_params(**{"verbose":self.verbose})
        except:
          pass
        print("\n..Reducing Dimensionality using: \n\t%s\n"%str(self.reducer))
        
      reducer = self.reducer
      X = reducer.fit_transform(X)
      
    # Scaling
    if self.scaler != None:
      if self.verbose > 0:
        print("\n..Scaling\n")
      scaler = self.scaler
      X = scaler.fit_transform(X)

    # We chop up the min-max column ranges into 'nr_cubes' parts
    self.chunk_dist = (np.max(X, axis=0) - np.min(X, axis=0))/self.nr_cubes

    # We calculate the overlapping windows distance 
    self.overlap_dist = self.overlap_perc * self.chunk_dist

    # We find our starting point
    self.d = np.min(X, axis=0)
    
    return X

  def map(self, X, dimension_index=[0], dimension_name=""):
    # This maps the data to a simplicial complex. Returns a dictionary with nodes and links.
    
    start = datetime.now()
    
    def cube_coordinates_all(nr_cubes, nr_dimensions):
      # if there are 4 cubes per dimension and 3 dimensions 
      # return the bottom left (origin) coordinates of 64 hypercubes, in a sorted list of Numpy arrays
      l = []
      for x in range(nr_cubes):
        l += [x] * nr_dimensions
      return [np.array(list(f)) for f in sorted(set(itertools.permutations(l,nr_dimensions)))]
    
    nodes = defaultdict(list)
    links = defaultdict(list)
    complex = {}
    
    if self.verbose > 0:
      print("Mapping on data shaped %s using dimensions %s\n"%(str(X.shape),str(dimension_index)))
    
    # Scaling
    if self.scaler != None:
      scaler = self.scaler
      X = scaler.fit_transform(X)
    
    # Initialize Cluster Algorithm
    clf = self.clf
    
    # Prefix'ing the data with ID's
    ids = np.array([x for x in range(X.shape[0])])
    X = np.c_[ids,X]

    # Subdivide the data X in intervals/hypercubes with overlap
    if self.verbose > 0:
      total_cubes = len(cube_coordinates_all(self.nr_cubes,len(dimension_index)))
      print("Creating %s hypercubes."%total_cubes)
    di = np.array(dimension_index)  
    for i, coor in enumerate(cube_coordinates_all(self.nr_cubes,di.shape[0])): 
      # Slice the hypercube
      hypercube = X[ np.invert(np.any((X[:,di+1] >= self.d[di] + (coor * self.chunk_dist[di])) & 
          (X[:,di+1] < self.d[di] + (coor * self.chunk_dist[di]) + self.chunk_dist[di] + self.overlap_dist[di]) == False, axis=1 )) ]
      
      if self.verbose > 1:
        print("There are %s points in cube_%s / %s with starting range %s"%
              (hypercube.shape[0],i,total_cubes,self.d[di] + (coor * self.chunk_dist[di])))
      
      # If at least one sample inside the hypercube
      if hypercube.shape[0] > 0:
        # Cluster the data point(s) inside the cube, skipping the id-column
        clf.fit(hypercube[:,1:])
        
        if self.verbose > 1:
          print("Found %s clusters in cube_%s\n"%(np.unique(clf.labels_[clf.labels_ > -1]).shape[0],i))
        
        #Now for every (sample id in cube, predicted cluster label)
        for a in np.c_[hypercube[:,0],clf.labels_]:
          if a[1] != -1: #if not predicted as noise
            cluster_id = str(coor[0])+"_"+str(i)+"_"+str(a[1])+"_"+str(coor)+"_"+str(self.d[di] + (coor * self.chunk_dist[di])) # Rudimentary cluster id
            nodes[cluster_id].append( int(a[0]) ) # Append the member id's as integers
      else:
        if self.verbose > 1:
          print("Cube_%s is empty.\n"%(i))

    # Create links when clusters from different hypercubes have members with the same sample id.
    for k in nodes:
      for kn in nodes:
        if k != kn:
          if len(nodes[k] + nodes[kn]) != len(set(nodes[kn] + nodes[k])): # there are non-unique id's in the union
            links[k].append( kn )
          
          # Create links between local hypercube clusters if setting link_local = True
          # This is an experimental feature deviating too much from the original mapper algo.
          # Creates a lot of spurious edges, and should only be used when mapping one or at most two dimensions.
          if self.link_local:
            if k.split("_")[0] == kn.split("_")[0]:
              links[k].append( kn )
        
    # Reporting
    if self.verbose > 0:
      nr_links = 0
      for k in links:
        nr_links += len(links[k])
      print("\ncreated %s edges and %s nodes in %s."%(nr_links,len(nodes),str(datetime.now()-start)))
    
    complex["nodes"] = nodes
    complex["links"] = links
    complex["meta"] = dimension_name

    return complex

  def visualize(self, complex, path_html="mapper_visualization_output.html", title="My Data", graph_link_distance=30, graph_gravity=0.1, graph_charge=-120, custom_tooltips=None, width_html=0, height_html=0, show_tooltips=True, show_title=True, show_meta=True):
    # Turns the dictionary 'complex' in a html file with d3.js
    
    # Format JSON
    json_s = {}
    json_s["nodes"] = []
    json_s["links"] = []
    k2e = {} # a key to incremental int dict, used for id's when linking

    for e, k in enumerate(complex["nodes"]):
      # Tooltip formatting
      if custom_tooltips != None:
        tooltip_s = "<h2>Cluster %s</h2>"%k + " ".join([str(f) for f in custom_tooltips[complex["nodes"][k]]])
        if self.color_function == "average_signal_cluster":
          tooltip_i = int(((sum([f for f in custom_tooltips[complex["nodes"][k]]]) / len(custom_tooltips[complex["nodes"][k]])) * 30) )
          json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(tooltip_i)})
        else:
          json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(k.split("_")[0])})
      else:
        tooltip_s = "<h2>Cluster %s</h2>Contains %s members."%(k,len(complex["nodes"][k]))
        json_s["nodes"].append({"name": str(k), "tooltip": tooltip_s, "group": 2 * int(np.log(len(complex["nodes"][k]))), "color": str(k.split("_")[0])})
      k2e[k] = e
    for k in complex["links"]:
      for link in complex["links"][k]:
        json_s["links"].append({"source": k2e[k], "target":k2e[link],"value":1})

    # Width and height of graph in HTML output
    if width_html == 0:
      width_css = "100%"
      width_js = 'document.getElementById("holder").offsetWidth-20'
    else:
      width_css = "%spx" % width_html
      width_js = "%s" % width_html
    if height_html == 0:
      height_css = "100%"
      height_js = 'document.getElementById("holder").offsetHeight-20'
    else:
      height_css = "%spx" % height_html
      height_js = "%s" % height_html
    
    # Whether to show certain UI elements or not
    if show_tooltips == False:
      tooltips_display = "display: none;"
    else:
      tooltips_display = ""
      
    if show_meta == False:
      meta_display = "display: none;"
    else:
      meta_display = ""
      
    if show_title == False:
      title_display = "display: none;"
    else:
      title_display = ""  
    
    with open(path_html,"wb") as outfile:
      html = """<!DOCTYPE html>
    <meta charset="utf-8">
    <meta name="generator" content="KeplerMapper">
    <title>%s | KeplerMapper</title>
    <link href='https://fonts.googleapis.com/css?family=Roboto:700,300' rel='stylesheet' type='text/css'>
    <style>
    * {margin: 0; padding: 0;}
    html { height: 100%%;}
    body {background: #111; height: 100%%; font: 100 16px Roboto, Sans-serif;}
    .link { stroke: #999; stroke-opacity: .333;  }
    .divs div { border-radius: 50%%; background: red; position: absolute; }
    .divs { position: absolute; top: 0; left: 0; }
    #holder { position: relative; width: %s; height: %s; background: #111; display: block;}
    h1 { %s padding: 20px; color: #fafafa; text-shadow: 0px 1px #000,0px -1px #000; position: absolute; font: 300 30px Roboto, Sans-serif;}
    h2 { text-shadow: 0px 1px #000,0px -1px #000; font: 700 16px Roboto, Sans-serif;}
    .meta {  position: absolute; opacity: 0.9; width: 220px; top: 80px; left: 20px; display: block; %s background: #000; line-height: 25px; color: #fafafa; border: 20px solid #000; font: 100 16px Roboto, Sans-serif;}
    div.tooltip { position: absolute; width: 380px; display: block; %s padding: 20px; background: #000; border: 0px; border-radius: 3px; pointer-events: none; z-index: 999; color: #FAFAFA;}
    }
    </style>
    <body>
    <div id="holder">
      <h1>%s</h1>
      <p class="meta">
      <b>Lens</b><br>%s<br><br>
      <b>Cubes per dimension</b><br>%s<br><br>
      <b>Overlap percentage</b><br>%s%%<br><br>
      <!-- <b>Linking locally</b><br>%s<br><br> -->
      <b>Color Function</b><br>%s( %s )<br><br>
      <b>Clusterer</b><br>%s<br><br>
      <b>Scaler</b><br>%s
      </p>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
    <script>
    var width = %s,
      height = %s;

    var color = d3.scale.ordinal()
      .domain(["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"])
      .range(["#FF0000","#FF1400","#FF2800","#FF3c00","#FF5000","#FF6400","#FF7800","#FF8c00","#FFa000","#FFb400","#FFc800","#FFdc00","#FFf000","#fdff00","#b0ff00","#65ff00","#17ff00","#00ff36","#00ff83","#00ffd0","#00e4ff","#00c4ff","#00a4ff","#00a4ff","#0084ff","#0064ff","#0044ff","#0022ff","#0002ff","#0100ff","#0300ff","#0500ff"]);

    var force = d3.layout.force()
      .charge(%s)
      .linkDistance(%s)
      .gravity(%s)
      .size([width, height]);

    var svg = d3.select("#holder").append("svg")
      .attr("width", width)
      .attr("height", height);
    
    var div = d3.select("#holder").append("div")   
      .attr("class", "tooltip")               
      .style("opacity", 0.0);
    
    var divs = d3.select('#holder').append('div')
      .attr('class', 'divs')
      .attr('style', function(d) { return 'overflow: hidden; width: ' + width + 'px; height: ' + height + 'px;'; });  
    
      graph = %s;

      force
        .nodes(graph.nodes)
        .links(graph.links)
        .start();

      var link = svg.selectAll(".link")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link")
        .style("stroke-width", function(d) { return Math.sqrt(d.value); });

      var node = divs.selectAll('div')
      .data(graph.nodes)
        .enter().append('div')
        .on("mouseover", function(d) {      
          div.transition()        
            .duration(200)      
            .style("opacity", .9);
          div .html(d.tooltip + "<br/>")  
            .style("left", (d3.event.pageX + 100) + "px")     
            .style("top", (d3.event.pageY - 28) + "px");    
          })                  
        .on("mouseout", function(d) {       
          div.transition()        
            .duration(500)      
            .style("opacity", 0);   
        })
        .call(force.drag);
      
      node.append("title")
        .text(function(d) { return d.name; });

      force.on("tick", function() {
      link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

      node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; })
        .attr('style', function(d) { return 'width: ' + (d.group * 2) + 'px; height: ' + (d.group * 2) + 'px; ' + 'left: '+(d.x-(d.group))+'px; ' + 'top: '+(d.y-(d.group))+'px; background: '+color(d.color)+'; box-shadow: 0px 0px 3px #111; box-shadow: 0px 0px 33px '+color(d.color)+', inset 0px 0px 5px rgba(0, 0, 0, 0.2);'})
        ;
      });
    </script>"""%(title,width_css, height_css, title_display, meta_display, tooltips_display, title,complex["meta"],self.nr_cubes,self.overlap_perc*100,self.link_local,self.color_function,complex["meta"],str(self.clf),str(self.scaler),width_js,height_js,graph_charge,graph_link_distance,graph_gravity,json.dumps(json_s))
      outfile.write(html.encode("utf-8"))
    if self.verbose > 0:
      print("\nWrote d3.js graph to '%s'"%path_html)