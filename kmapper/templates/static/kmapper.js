// Height and width settings
var canvas_height = window.innerHeight - 5;
document.getElementById("canvas").style.height = canvas_height + "px";
var width = document.getElementById("canvas").offsetWidth;
var height = document.getElementById("canvas").offsetHeight;
var w = width;
var h = height;
var padding = 40;

var focus_node = null, highlight_node = null;
var text_center = false;
var outline = false;

// Size for zooming
var size = d3.scale.pow().exponent(1)
           .domain([1,100])
           .range([8,24]);

// Variety of variable inits
var highlight_color = "blue";
var highlight_trans = 0.1;
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


var tocolor = "fill";
var towhite = "stroke";
if (outline) {
  tocolor = "stroke";
  towhite = "fill";
}


// We draw the graph in SVG
var svg = d3.select("#canvas").append("svg")
          .attr("width", width)
          .attr("height", height);

svg.style("cursor","move");
var g = svg.append("g");

/**
 * Side panes
 * 
 * 
 * 
 * 
 */

// Show/Hide Functionality
var toggle_pane = function(content, content_id, tag){
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


/**
 *  Graph setup 
 * 
 * 
 */

var graph = JSON.parse(document.getElementById("json_graph").dataset.graph);

              
// Force settings
var force = d3.layout.force()
            .linkDistance(5)
            .gravity(0.2)
            .charge(-1200)
            .size([w,h]);

force
  .nodes(graph.nodes)
  .links(graph.links)
  .start();

var link = g.selectAll(".link")
            .data(graph.links)
            .enter().append("line")
              .attr("class", "link")
              .style("stroke-width", function(d) { return d.w * nominal_stroke; })
              .style("stroke-width", function(d) { return d.w * nominal_stroke; })


var node = g.selectAll(".node")
            .data(graph.nodes)
            .enter().append("g")
              .attr("class", "node")
              .call(force.drag);

// Double clicking on a node will center on it.
node.on("dblclick.zoom", function(d) { d3.event.stopPropagation();
  var dcx = (window.innerWidth/2-d.x*zoom.scale());
  var dcy = (window.innerHeight/2-d.y*zoom.scale());
  zoom.translate([dcx,dcy]);
  g.attr("transform", "translate("+ dcx + "," + dcy  + ")scale(" + zoom.scale() + ")");
});




// Drop-shadow Filter
var svg = d3.select("svg");
var defs = svg.append("defs");
var dropShadowFilter = defs.append('svg:filter')
  .attr('id', 'drop-shadow')
  .attr('filterUnits', "userSpaceOnUse")
  .attr('width', '250%')
  .attr('height', '250%');
dropShadowFilter.append('svg:feGaussianBlur')
  .attr('in', 'SourceGraphic')
  .attr('stdDeviation', 12)
  .attr('result', 'blur-out');
dropShadowFilter.append('svg:feColorMatrix')
  .attr('in', 'blur-out')
  .attr('type', 'hueRotate')
  .attr('values', 0)
  .attr('result', 'color-out');
dropShadowFilter.append('svg:feOffset')
  .attr('in', 'color-out')
  .attr('dx', 0)
  .attr('dy', 0)
  .attr('result', 'the-shadow');
dropShadowFilter.append('svg:feComponentTransfer')
  .attr('type', 'linear')
  .attr('slope', 0.2)
  .attr('result', 'shadow-opacity');
dropShadowFilter.append('svg:feBlend')
  .attr('in', 'SourceGraphic')
  .attr('in2', 'the-shadow')
  .attr('mode', 'normal');

// Draw circles
var circle = node.append("path")
  .attr("d", d3.svg.symbol()
    .size(function(d) { return d.size * 50; })
    .type(function(d) { return d.type; }))
  .attr("class", "circle")
  .style(tocolor, function(d) {
    console.log("Node color:", d.color);
    console.log("becomes color ", color(d.color));
    return color(d.color);
  });
//.style("filter", "url(#drop-shadow)");


// Format all text 
var text = g.selectAll(".text")
  .data(graph.nodes)
  .enter().append("text")
    .attr("dy", ".35em")
    .style("font-family", "Roboto")
    .style("font-weight", "400")
    .style("color", "#2C3E50")
    .style("font-size", nominal_text_size + "px");



if (text_center) {
  text.text(function(d) { return d.id; })
    .style("text-anchor", "middle");
} else {
  text.attr("dx", function(d) {return (size(d.size)||nominal_base_node_size);})
    .text(function(d) { return '\u2002'+d.id; });
}


/**
 * Mouse Interactivity
 * 
 * 
 * 
 * 
 * 
 */

node.on("mouseover", function(d) {
  // Change node details
  set_highlight(d);
  d3.select("#tooltip_content").html(d.tooltip + "<br/>");
}).on("mousedown", function(d) {
  // TODO: This seems to only stop the one particular node from moving?

  d3.event.stopPropagation();
  focus_node = d;
  if (highlight_node === null) {
    set_highlight(d)
  }
}).on("mouseout", function(d) {
  exit_highlight();
});

d3.select(window).on("mouseup", function() {
  if (focus_node!==null){
    focus_node = null;
  }
  if (highlight_node === null) {
    exit_highlight();
  }
});

// Node highlighting logic
function exit_highlight(){
  highlight_node = null;
  if (focus_node===null){
    svg.style("cursor","move");
  }
}

function set_highlight(d){
  svg.style("cursor","pointer");
  if (focus_node!==null) d = focus_node;
}


// Zoom logic
zoom.on("zoom", function() {
  var stroke = nominal_stroke;
  var base_radius = nominal_base_node_size;
  if (nominal_base_node_size*zoom.scale()>max_base_node_size) {
    base_radius = max_base_node_size/zoom.scale();}
  circle.attr("d", d3.svg.symbol()
    .size(function(d) { return d.size * 50; })
    .type(function(d) { return d.type; }))
  if (!text_center) text.attr("dx", function(d) {
    return (size(d.size)*base_radius/nominal_base_node_size||base_radius); });

  var text_size = nominal_text_size;
  if (nominal_text_size*zoom.scale()>max_text_size) {
    text_size = max_text_size/zoom.scale(); }
  text.style("font-size",text_size + "px");

  g.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
});

svg.call(zoom);
resize();
d3.select(window).on("resize", resize);

// Animation per tick
force.on("tick", function() {
  node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  text.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  link.attr("x1", function(d) { return d.source.x; })
    .attr("y1", function(d) { return d.source.y; })
    .attr("x2", function(d) { return d.target.x; })
    .attr("y2", function(d) { return d.target.y; });
  node.attr("cx", function(d) { return d.x; })
    .attr("cy", function(d) { return d.y; });
});



// Resizing window and redraws
function resize() {
  var width = window.innerWidth, height = window.innerHeight;
  var width = document.getElementById("canvas").offsetWidth;
  var height = document.getElementById("canvas").offsetHeight;
  svg.attr("width", width).attr("height", height);

  force.size([force.size()[0]+(width-w)/zoom.scale(),
              force.size()[1]+(height-h)/zoom.scale()]).resume();
  w = width;
  h = height;
}


function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

// Key press events
window.addEventListener("keydown", function (event) {
if (event.defaultPrevented) {
  return; // Do nothing if the event was already processed
}


switch (event.key) {
  case "f":
    console.log("Freeze graph")
    break;
  case "s":
    // Do something for "s" key press.
    node.style("filter", "url(#drop-shadow)");
    break;
  case "c":
    // Do something for "s" key press.
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
// Cancel the default action to avoid it being handled twice
event.preventDefault();
}, true);
