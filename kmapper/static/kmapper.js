// Height and width settings
var canvas_height = window.innerHeight - 5;
document.getElementById("canvas").style.height = canvas_height + "px";
var width = document.getElementById("canvas").offsetWidth;
var height = document.getElementById("canvas").offsetHeight;
var w = width;
var h = height;
var padding = 40;

var focus_node = null;
var text_center = false;
var outline = false;

// Size for zooming
var size = d3.scale.pow().exponent(1)
           .domain([1,100])
           .range([8,24]);

// Variety of variable inits
var default_node_color = "#ccc";
var default_node_color = "rgba(160,160,160, 0.5)";
var default_link_color = "rgba(160,160,160, 0.5)";
var nominal_base_node_size = 8;
var nominal_text_size = 15;
var max_text_size = 24;
var nominal_stroke = 1.0;
var max_stroke = 4.5;
var max_base_node_size = 36;
var min_zoom = 0.1;
var max_zoom = 7;
var zoom = d3.behavior.zoom().scaleExtent([min_zoom,max_zoom]);
var svg, g;
var force;
var link, node;
var drag;
var dragging = false;
var circle;
var text;
var focus_via_click = false;
var nodes = [];
var links = [];

var tocolor = "fill";
var towhite = "stroke";
if (outline) {
  tocolor = "stroke";
  towhite = "fill";
}

/**
 * Side panes
 */

// Show/Hide Functionality
function toggle_pane(content, content_id, tag) {
  var active = content.active ? false : true;

  if (active){
    content_id.style("display", "unset");
    tag.textContent = "[-]";
  } else{
    content_id.style("display", "none");
    tag.textContent = "[+]";
  }

  // TODO: This is probably not the best way to find the correct height.
  var h = canvas_height - content.offsetTop - padding;
  content_id.style("height", h + "px")

  content.active = active;
}

d3.select("#tooltip_control").on("click", function() {
  toggle_pane(tooltip_content,
              d3.select("#tooltip_content"),
              d3.select("#tooltip_tag")[0][0]);

});

d3.select("#meta_control").on("click", function() {
  toggle_pane(meta_content,
              d3.select("#meta_content"),
              d3.select("#meta_tag")[0][0])

});

d3.select("#help_control").on("click", function() {
  toggle_pane(helptip_content,
              d3.select("#helptip_content"),
              d3.select("#helptip_tag")[0][0])
});

/**
 *
 * Set up color scale
 *
 *
 */
var colorscale = JSON.parse(document.getElementById("json_colorscale").dataset.colorscale);
var domain = colorscale.map((x)=>x[0])
var palette = colorscale.map((x)=>x[1])

var color = d3.scale.linear()
  .domain(domain)
  .range(palette);

var graph = JSON.parse(document.getElementById("json_graph").dataset.graph);

/*
* one-time setups, like SVG and force init
*/
function init() {
  // We draw the graph in SVG
  svg = d3.select("#canvas svg")
          .attr("width", width)
          .attr("height", height)
          .style("cursor","move")
          .call(zoom)
          .on('mousedown.focus', function(e){
            set_focus_via_click(null);
          })

  g = svg.append("g");

  link = g.selectAll(".link")
  node = g.selectAll(".node")
  text = g.selectAll(".text")

  force = d3.layout.force()
    .nodes(nodes)
    .links(links)
    .linkDistance(5)
    .gravity(0.2)
    .charge(-1200)
    .size([w,h])
    .on('tick', tick);

  drag = force.drag()
              .on("dragstart", function(d){
                svg.style('cursor','grabbing');
                d.fixed = true;
                dragging = true;
              })
              .on('dragend', function(d){
                dragging = false;
              });

  resize();
  d3.select(window).on("resize", resize);

  d3.select(window).on("mouseup.focus", function(){
    if (focus_node != null) {
      set_cursor('pointer');
    }
    if (focus_node == null) {
      set_cursor('move');
    }
  });

  // Zoom logic
  zoom.on("zoom", function() {
    var stroke = nominal_stroke;
    var base_radius = nominal_base_node_size;
    if (nominal_base_node_size*zoom.scale()>max_base_node_size) {
      base_radius = max_base_node_size/zoom.scale();
    }

    circle.attr("d", d3.svg.symbol()
      .size(function(d) { return d.size * 50; })
      .type(function(d) { return d.type; }))

    if (!text_center) text.attr("dx", function(d) {
      return (size(d.size)*base_radius/nominal_base_node_size||base_radius);
    });

    var text_size = nominal_text_size;
    if (nominal_text_size*zoom.scale()>max_text_size) {
      text_size = max_text_size/zoom.scale();
    }
    text.style("font-size",text_size + "px");

    g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
  });
}

function tick() {
  node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
      .attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; })

  text.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

  link.attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });
}

/*
* Resizing window and redraws
*/
function resize() {
  var width = document.getElementById("canvas").offsetWidth;
  var height = document.getElementById("canvas").offsetHeight;

  svg.attr("width", width)
     .attr("height", height);

  force.size([
      force.size()[0]+(width-w)/zoom.scale(),
      force.size()[1]+(height-h)/zoom.scale()
    ]).resume();

  w = width;
  h = height;
}

function set_focus_via_click(d) {
  focus_via_click = (d != null ? true : false);
  set_focus_node(d);
}

function set_focus_node(d){
  if (d == null) {
    focus_node = null;
    set_cursor('move');
    d3.select("#tooltip_content").html('');
    exit_highlight();
  } else if (d.name != focus_node) {
    exit_highlight(focus_node);
    focus_node = d.name;
    set_highlight(focus_node);
    set_cursor('pointer');
    d3.select("#tooltip_content").html(d3.select("#node_tooltip_data-" + d.tooltip.node_id).html());
  }
  // else, it's already the focus node, so do nothing...
}

function set_highlight(node_id) {
  d3.select('#node-' + node_id + ' .circle').classed('highlight', true);
  d3.select('#node-' + node_id).classed('highlight', true);
}

function exit_highlight(node_id) {
  if (!node_id) {
     d3.selectAll('.node .circle').classed('highlight', false);
     d3.selectAll('.node').classed('highlight', false);
  } else {
     d3.select('#node-' + node_id + ' .circle').classed('highlight', false);
     d3.select('#node-' + node_id).classed('highlight', false);
  }
}

function set_cursor(state) {
  if (!dragging) {
    svg.style('cursor', state);
  }
}

// https://bl.ocks.org/mbostock/3750558
// IIRC I copied this from the source for what d3 force layout normally does
// on dragStart and dragEnd.
// Sets (unsets) the third bit to 1 if doing a mouseover (mouseout).
function d3_layout_forceMouseover(d) {
  d.fixed |= 4;
  d.px = d.x, d.py = d.y;
}
function d3_layout_forceMouseout(d) {
  d.fixed &= ~4;
}

// Double clicking on a node will center on it.
function node_dblclick(d) {
  d3.event.stopPropagation();
  var dcx = ( window.innerWidth/2 - d.x * zoom.scale() );
  var dcy = ( window.innerHeight/2 - d.y * zoom.scale() );
  zoom.translate([dcx, dcy]);
  g.attr("transform", "translate("+ dcx + "," + dcy  + ")scale(" + zoom.scale() + ")");
}

function node_mouseover(d) {
  // Change node details
  d3_layout_forceMouseover(d);
  if (d3.event.buttons == 0){
    set_cursor('pointer');
    if (!focus_via_click) {
      set_focus_node(d);
    }
  }
}

function node_mouseout(d) {
  d3_layout_forceMouseout(d);
  if (d3.event.buttons == 0){
    set_cursor('move');
    if (!focus_via_click) {
        set_focus_node(null);
    }
  }
}

function node_mousedown(d) {
  d3.event.stopPropagation();
  if (focus_node != d.name) {
      //switch click focus
      set_focus_via_click(d);
  } else if (!focus_via_click) {
      //d already selected but not via click; set click true
      focus_via_click = true;
  }
}

function start() {
  // shallow copy to enable restarting,
  // because calling force.start() mutates the links (replaces indeces with refs)
  nodes = graph.nodes.map(n => Object.assign({}, n));
  links = graph.links.map(l => Object.assign({}, l));

  force
    .nodes(nodes)
    .links(links);

  link = link.data(links);
  link.enter().append("line")
        .attr("class", "link")
        .style("stroke-width", function(d) { return d.w * nominal_stroke; })
        .style("stroke-width", function(d) { return d.w * nominal_stroke; });
  link.exit().remove()

  node = node.data(nodes);

  node.enter().append("g")
        .attr("class", "node")
        .attr("id", function(d){ return "node-" + d.name })
        .call(drag);
  node.exit().remove()

  // Draw circles
  circle = node.append("path")
    .attr("d", d3.svg.symbol()
      .size(function(d) { return d.size * 50; })
      .type(function(d) { return d.type; }))
    .attr("class", "circle")
    .style(tocolor, function(d) { return color(d.color); });

  // bind after circle appends
  node.on("mouseover.focus", node_mouseover)
      .on("mouseout.focus", node_mouseout)
      .on('mousedown.focus', node_mousedown)
      .on("dblclick.zoom", node_dblclick)

  // Format all text
  text = text.data(nodes);
  text.enter().append("text")
      .attr("dy", ".35em")
      .style("font-family", "Roboto")
      .style("font-weight", "400")
      .style("color", "#2C3E50")
      .style("font-size", nominal_text_size + "px");
  text.exit().remove();

  if (text_center) {
    text.text(function(d) { return d.id; })
        .style("text-anchor", "middle");
  } else {
    text.attr("dx", function(d) { return ( size(d.size) || nominal_base_node_size ); })
        .text(function(d) { return '\u2002' + d.id; });
  }

  force.start();
}

init();
start();

function restart() {
  // node.remove()
  // link.remove()
  focus_via_click = false;
  start()
}

function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

// Key press events
d3.select(window).on("keydown", function () {
  if (d3.event.defaultPrevented) {
    return; // Do nothing if the event was already processed
  }

  if (!d3.event.ctrlKey && !d3.event.altKey && !d3.event.metaKey) {
    switch (d3.event.key) {
      case "f": // freeze all
        node.datum(function(d){ d.fixed = true; return d; });
        break;
      case "x": // unfreeze all
        node.datum(function(d){ d.fixed = false; return d; });
        force.resume()
        break
      case "s":
        // Glow
        node.style("filter", "url(#drop-shadow-glow)");
        break;
      case "c":
        // Remove glow
        node.style("filter", null);
        break;
      case "p":
        // Turn to print mode, white backgrounds
        d3.select("body").attr('id', null).attr('id', "print")
        break;
      case "d":
        // Do something for "d" key press.
        d3.select("body").attr('id', null).attr('id', "display")
        break;
      case "z":
        force.gravity(0.0)
             .charge(0.0);
        resize();
        break
      case "m":
        force.gravity(0.07)
             .charge(-1);
        resize();
        break
      case "e":
        force.gravity(0.4)
             .charge(-600);
        resize();
        break
      default:
        return; // Quit when this doesn't handle the key event.
    }
  d3.event.preventDefault();
  }
  // Cancel the default action to avoid it being handled twice
}, true);

/*
* Save and load config
*
*/

// save config
document.getElementById('download-config').addEventListener('click', function(){
  let config = {}
  node.data().forEach(node => {
      let config_node = {}
      config_node['fixed'] = 'fixed' in node && !!node['fixed']
      config_node['x'] = config_node['px'] = node['x']
      config_node['y'] = config_node['py'] = node['y']
      config[node['name']] = config_node
    })

  //JSON.stringify(config,undefined,2)

  // https://stackoverflow.com/a/45594892
  var fileName = 'kmapper-config.json';

  // Create a blob of the data. Blob is native JS api
  var fileToSave = new Blob([JSON.stringify(config)], {
      type: 'application/json',
      name: fileName
  });

  // function from FileSaver.js
  saveAs(fileToSave, fileName);
})

// load config
var config_file_loader = document.getElementById('config-file-loader');

config_file_loader.addEventListener('change', function(){
  document.getElementById('load-config').disabled = ( config_file_loader.files.length === 0 )
})

document.getElementById('load-config').addEventListener('click', function(){
  const config_file = config_file_loader.files[0];
  const fr = new FileReader();
  fr.onload = function(e) {
    var config = JSON.parse(e.target.result);
    load_config(config);

  }
  fr.readAsText(config_file)
})
function load_config(config){
  node = node.datum(function(d, i){
    let load_node_config = config[d['name']]
    d = Object.assign(d, load_node_config);
    return d;
  })
  force.resume()
}
