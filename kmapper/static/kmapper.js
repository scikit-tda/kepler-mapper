// Height and width settings
var page_height = window.innerHeight - 5;
var header_height = document.getElementById('header').offsetHeight;
var canvas_height = page_height - header_height;
document.getElementById("canvas").style.height = canvas_height + "px";
var width = document.getElementById("canvas").offsetWidth;
var height = document.getElementById("canvas").offsetHeight;
var w = width;
var h = height;
var padding = 40;

var focus_node_id = null;
var focus_node = null;
var text_center = false;
var outline = false;

// Size for zooming
var size = d3.scalePow().exponent(1)
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
var zoom;
var svg, g;
var simulation;
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

  if (active) {
    content_id.style("display", "block");
    tag.node().textContent = "[-]";
  } else {
    content_id.style("display", "none");
    tag.node().textContent = "[+]";
  }

  // TODO: This is probably not the best way to find the correct height.
  var h = canvas_height - content.offsetTop - padding;
  content_id.style("height", h + "px")

  content.active = active;
}

d3.select("#tooltip_control").on("click", function(e) {
  toggle_pane(tooltip_content,
              d3.select("#tooltip_content"),
              d3.select("#tooltip_tag"))

});

d3.select("#meta_control").on("click", function(e) {
  toggle_pane(meta_content,
              d3.select("#meta_content"),
              d3.select("#meta_tag"))

});

d3.select("#help_control").on("click", function(e) {
  toggle_pane(helptip_content,
              d3.select("#helptip_content"),
              d3.select("#helptip_tag"))
});

d3.select('#select-color-function').on('input', function(e){
  color_function_index = parseInt(e.target.value);
  update_color_functions()
})
d3.select('#select-node-color-function').on('input', function(e){
  node_color_function_index = parseInt(e.target.value);
  update_color_functions()
})

/**
 *
 * Set up color scale
 *
 *
 */
// var colorscale defined in base.html
var domain = colorscale.map((x)=>x[0])
var palette = colorscale.map((x)=>x[1])

var color = d3.scaleLinear()
  .domain(domain)
  .range(palette);

/*
* one-time setups, like SVG and force init
*/
function init() {

  zoom = d3.zoom()
    .scaleExtent([min_zoom, max_zoom])
    .on('zoom', zoomed)

  // We draw the graph in SVG
  svg = d3.select("#canvas svg")
          .attr("width", width)
          .attr("height", height)
          .style("cursor","move")
          .call(zoom)
          .on('dblclick.zoom', null); // prevent default zoom-in on dblclick

  svg.on('click.focus', function(e){
    set_focus_via_click(null);
  });

  g = svg.append("g")

  link = g.selectAll(".link")
  node = g.selectAll(".node")
  text = g.selectAll(".text")

  simulation = d3.forceSimulation()
    .force('charge', d3.forceManyBody().strength(-1200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('link', d3.forceLink().distance(5))
    .force('x', d3.forceX()) // not sure what this does...
    .force('y', d3.forceY())
    .on('tick', ticked)

  drag = d3.drag()
    .on("start", function(e, d){
      svg.style('cursor','grabbing');
      if (!e.active) {
        simulation.alphaTarget(0.3).restart()
      }
      d.fx = d.x
      d.fy = d.y
      dragging = true;
    })
    .on('drag', function(e, d){
      d.fx = e.x
      d.fy = e.y
    })
    .on('end', function(e, d){
      if (!e.active) {
        simulation.alphaTarget(0)
      }
      dragging = false;
    });

  resize();
  d3.select(window).on("resize", resize);

  d3.select(window).on("mouseup.focus", function(e){
    if (focus_node == null) {
      set_cursor('move');
    } else {
      set_cursor('pointer');
    }
  });
}

function set_histogram(selection, data){
  selection.selectAll('.bin')
    .data(data)
    .join(
      enter => enter.append('div')
        .attr('class', 'bin')
        .call(enter => enter.append('div')
          .text(d => d.perc + '%'))
      ,
      update => update
        .call(update => update.select('div')
          .text(d => d.perc + '%'))
    )
    .style('height', (d) => (d.height || 1) + 'px')
    .style('background', (d) => d.color);
}



var color_function_index = 0;
var node_color_function_index = 0;

function update_color_functions(){
  // update_meta_content_histogram
  set_histogram(d3.select('#meta_content .histogram'), summary_histogram[node_color_function_index][color_function_index])

  // update node colors
  node.style(tocolor, function(d) {
    return color(d.color[node_color_function_index][color_function_index]);
  })

  // update focus node display, if focus_node
  if (focus_node != null){
    set_focus_node_histogram(focus_node)
  }
}

function update_meta_content_histogram(){

}

function draw_circle_size(d) {
  return (d3.symbol()
    .size(function(d) {
      if (!d.size_modifier) {
          d.size_modifier = 1;
      }
      return d.size * 50 * d.size_modifier;
    })
    .type(d3.symbolCircle))(d)
}

function start() {

  /*
  * Force-related things
  *
  */
  // shallow copy to enable restarting,
  // because otherwise, starting the force simulation mutates the links (replaces indeces with refs)
  nodes = graph.nodes.map(n => Object.assign({}, n));
  links = graph.links.map(l => Object.assign({}, l));

  // draw links first so that they appear behind nodes
  link = link
    .data(links)
    .join("line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return d.w * nominal_stroke; })
      .style("stroke-width", function(d) { return d.w * nominal_stroke; });

  node = node
    .data(nodes, d => d.name)
    .join(enter => enter.append("g")
      .attr("class", "node")
      .attr("id", function(d){ return "node-" + d.name })
      // append circles...
      .append("path")
        .attr("d", draw_circle_size )
        .attr("class", "circle")
        .on("mouseover.focus", node_mouseover)
        .on("mouseout.focus", node_mouseout)
        .on('mousedown.focus', node_mousedown)
        .on("dblclick.freeze", (e, d) => unfreeze_node(d) )
        .on('click.zoom', node_click)
        .on('center_viewport', center_on_node)
        .call(drag));

  simulation.nodes(nodes);
  simulation.force('link').links(links);
  simulation.alpha(1).restart()

  update_color_functions()
}

init();
start();

function restart() {
  // nodes = []
  // links = []
  // node.remove()
  // link.remove()
  focus_via_click = false;
  start()
}

function zoomed({transform}){
  g.attr('transform', transform)
}

function ticked() {
  node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
      .attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; })

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

  simulation.force('center', d3.forceCenter(width / 2, height / 2));
  simulation.alpha(.3).restart();

  w = width;
  h = height;
}

function set_focus_via_click(d) {
  focus_via_click = (d != null ? true : false);
  set_focus_node(d);
}

function do_above_below_stats(wrapper, data) {
  let column_order = ['feature', 'mean', 'std'];
  wrapper.select('tbody')
    .selectAll('tr')
    .data(data, d => d)
    .join(enter => {
        let tr = enter.append('tr')
        tr.append('td')
        tr.append('td').style('font-size', 'smaller')
        tr.append('td').style('font-size', 'smaller').attr('class', 'std')
        return tr
        })
    .selectAll('td')
    .text( (d, i) => d[column_order[i]] )
  if (data.length > 0) {
    wrapper.style('display', 'block')
  } else {
    wrapper.style('display', 'none')
  }
}

function do_projection_stats(wrapper, data) {
  let column_order = ['name', 'mean', 'min', 'max'];
  wrapper.select('tbody').selectAll('tr')
    .data(data, d => d)
    .join(enter => {
      let tr = enter.append('tr')
      tr.append('td')
      tr.append('td').style('font-size', 'smaller')
      tr.append('td').style('font-size', 'smaller')
      tr.append('td').style('font-size', 'smaller')
      return tr;
    })
    .selectAll('td')
    .text( (d, i) => d[column_order[i]] );
  if (data.length > 0) {
    wrapper.style('display', 'block')
  } else {
    wrapper.style('display', 'none')
  }
}

let focus_node_tooltip_select = d3.select('#tooltip_content_focus_node');
let tooltip_content_no_focus_node = d3.select('#tooltip_content_no_focus_node')
let projection_stats_select = focus_node_tooltip_select.select('.projection_stats');
let cluster_stats_select = focus_node_tooltip_select.select('.cluster_stats');
let above_wrapper = cluster_stats_select.select('.above-wrapper');
let below_wrapper = cluster_stats_select.select('.below-wrapper');
let list_of_members = focus_node_tooltip_select.select('.membership-information .list-of-members');

function set_focus_node(d){
  if (d == null) {
    focus_node = null;
    set_cursor('move');
    tooltip_content_no_focus_node.style('display', 'block');
    focus_node_tooltip_select.style('display', 'none')
    exit_highlight();
  } else if (focus_node == null || d.name != focus_node.name) {
    exit_highlight(focus_node);
    focus_node = d;
    set_highlight(focus_node);
    set_cursor('pointer');

    tooltip_content_no_focus_node.style('display', 'none');

    focus_node_tooltip_select.select('button.center-on-node').node().dataset.nodeId = d.tooltip.node_id;
    focus_node_tooltip_select.select('.node_id').text(d.tooltip.node_id)
    focus_node_tooltip_select.select('.distribution_label').text(d.tooltip.dist_label)

    // histogram
    set_focus_node_histogram(d)

    // projection statistics
    let projection_stats = d.tooltip.projection_stats;
    do_projection_stats(projection_stats_select, projection_stats)

    // cluster statistics
    let cluster_stats = d.tooltip.cluster_stats;
    if (cluster_stats) {

      do_above_below_stats(above_wrapper, cluster_stats.above);

      do_above_below_stats(below_wrapper, cluster_stats.below);

      cluster_stats_select.select('.cluster-size').text(cluster_stats.size)

      cluster_stats_select.style('display', 'block')
    } else {
      cluster_stats_select.style('display', 'none')
    }

    // membership information
    list_of_members.selectAll('span')
      .data(d.tooltip.custom_tooltips, d => d)
      .join('span')
        .html(d => d)
        .style('display', 'inline-block')
        .style('padding', '0 3px')

    focus_node_tooltip_select.style('display', 'block');

    set_focus_node_histogram(d)
  }
  // else, it's already the focus node, so do nothing...
}

d3.select('#tooltip .center-on-node').on('click', function(e){
  // d3.select('#node-' + focus_node.name + ' .circle').dispatch('center_viewport')
  d3.select('#node-' + e.target.dataset.nodeId + ' .circle').dispatch('center_viewport')
})

function set_focus_node_histogram(d){
  set_histogram(d3.select('#tooltip_content .histogram'), d.tooltip.histogram[color_function_index])
}

function set_highlight(node) {
  let node_id = node.name;
  d3.select('#node-' + node_id + ' .circle').classed('highlight', true);
  d3.select('#node-' + node_id).classed('highlight', true);
}

function exit_highlight(node) {
  let node_id = false;
  if (node) {
    node_id = node.name
  }
  if (!node_id) {
     d3.selectAll('.node .circle').classed('highlight', false);
     d3.selectAll('.node').classed('highlight', false);
  } else {
     d3.select('#node-' + node_id + ' .circle').classed('highlight', false);
     d3.select('#node-' + node_id).classed('highlight', false);
  }

  do_projection_stats(projection_stats_select, [])
  do_above_below_stats(above_wrapper, [])
  do_above_below_stats(below_wrapper, [])
}

function set_cursor(state) {
  if (!dragging) {
    svg.style('cursor', state);
  }
}

function node_is_fixed(d){
  return d.hasOwnProperty('fx') || d.hasOwnProperty('fy')
}

function node_click(e, d) {
  e.stopPropagation()
  // to prevent the svg click.focus listener from unsetting the focus node...
}

function center_on_node(e, d) {
  svg.transition().duration(250).call(zoom.translateTo, d.x, d.y);
}

function node_mouseover(e, d) {
  d.was_fixed = d.hasOwnProperty('fx') || d.hasOwnProperty('fy')
  if (!d.was_fixed){
    d.fx = d.x;
    d.fy = d.y;
  }

  if (e.buttons == 0){
    set_cursor('pointer');
    if (!focus_via_click) {
      set_focus_node(d);
    }
  }
}

function node_mouseout(e, d) {
  if (!d.was_fixed){
      delete d.fx
      delete d.fy
  }
  if (e.buttons == 0){
    set_cursor('move');
    if (!focus_via_click) {
        set_focus_node(null);
    }
  }
}

function node_mousedown(e, d) {
  e.stopPropagation();
  if (focus_node.name != d.name) {
      //switch click focus
      set_focus_via_click(d);
  } else if (!focus_via_click) {
      //d already selected but not via click; set click true
      focus_via_click = true;
  }
}



function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

function unfreeze_node(d){
  delete d.fy
  delete d.fx
  return d
}

function freeze_node(d){
  d.fx = d.x;
  d.fy = d.y;
  return d
}

// search functionality
d3.select('#searchbar')
  .on('submit', function(event){
      /*
      * Searchbar functionality
      *
      * Permits AND, OR, and EXACT functionality
      *
      */
      event.preventDefault();
      // always running this will clear search results on a submit with an empty query
      node.datum(d => { d.size_modifier = 1; return d });
      node.style('filter', null)

      node.datum(d => {
        let to_lower = tooltip => String(tooltip).toLowerCase();
        d.tooltip.custom_tooltips_lowercase = d.tooltip.custom_tooltips.map(to_lower)
        return d
      });

      let search_query = d3.select(this).select('input').property('value').toLowerCase();
      if (search_query) {
          let search_mode = d3.select(this).select('input[name="search_mode"]:checked').property('value');

          let node_ratio_fn = (d, i) => {
            matches = d.tooltip.custom_tooltips_lowercase.map(map_fn)
            let how_many = matches.filter(x=>x).length;

            // Future optional feature -- size relative to ratio _within-node_
            let out_of = d.tooltip.cluster_stats.size;
            let ratio = how_many / out_of;

            // Node sizes will be overall number of items in the node
            // number of matching
            d.size_modifier = how_many;
          }

          let map_fn;
          let node_each_fn;
          switch (search_mode) {
            case 'and':
              search_query = search_query.split(' ');
              map_fn = tooltip => {
                return search_query.every(query_word => { return tooltip.includes( query_word ) })
              }
              node.each(node_ratio_fn)
              break;
            case 'or':
              search_query = search_query.split(' ');
              map_fn = tooltip => {
                return search_query.some(query_word => { return tooltip.includes( query_word ) })
              }
              node.each(node_ratio_fn)
              break;
            case 'exact':
              node.filter((d,i) => {
                  matches = d.tooltip.custom_tooltips_lowercase.map(tooltip => tooltip == search_query)
                  return matches.some(e => e);
              })
              .style("filter", "url(#drop-shadow-glow)");
              break;
            default:
              console.error(`search mode ${search_mode} unknown`);
              return;
          }

      }
      node.attr("d", draw_circle_size )

  })


//https://stackoverflow.com/questions/51319147/map-default-value
class MapWithDefault extends Map {
  get(key) {
    if (!this.has(key)) {
      this.set(key, this.default())
    };
    return super.get(key);
  }

  constructor(defaultFunction, entries) {
    super(entries);
    this.default = defaultFunction;
  }
}

d3.select('#min_intersction_selector')
  .on('submit', function(event){
    // replicates the logic in kmapper.nerve.GraphNerve.compute
    event.preventDefault()

    let result = new MapWithDefault(() => []);

    // loop over all combinations of nodes
    // https://stackoverflow.com/a/43241295/1396649
    let candidates = []
    let num_nodes = graph.nodes.length;
    for (let i = 0; i < num_nodes - 1; i++){
      for (let j = i + 1; j < num_nodes; j++) {
        candidates.push([i, j]);
      }
    }

    candidates.forEach(function(candidate) {
      let node1_idx = candidate[0];
      let node2_idx = candidate[1];
      let node1 = graph.nodes[node1_idx];
      let node2 = graph.nodes[node2_idx];
      let intersection = node1.tooltip.custom_tooltips.filter(x => node2.tooltip.custom_tooltips.includes(x));
      if (intersection.length >= Number(min_intersction_selector_input.property('value')) ) {
        result.get(node1_idx).push(node2_idx)
      }
    })

    let edges = []
    result.forEach(function(value, key) {
      let _edges = value.map(function(end) {
        return [key, end]
      })
      edges.push(_edges);
    })

    edges = edges.flat().map(function(edge) {
      return {
        'source': edge[0],
        'target': edge[1],
        'width': 1
      }
    })

    graph.links = edges;
    restart()
  })


// Dynamically size the min_intersection_selector input
let min_intersction_selector_input = d3.select('#min_intersction_selector input');

min_intersction_selector_input
  .on('input', function(event){
    resizeInput.call(this)
  })

function resizeInput() {
  this.style.width = ( this.value.length + 3 ) + "ch";
}

// Only trigger if input is present
if (min_intersction_selector_input.size()) {
  resizeInput.call(min_intersction_selector_input.node())
}


// Key press events
let searchbar = d3.select('#searchbar input');

d3.select(window).on("keydown", function (event) {
  if (event.defaultPrevented) {
    return; // Do nothing if the event was already processed
  }

  // if searchbar is present and has focus
  if (searchbar.size() && searchbar.node().matches(':focus')){
      return; // let them use the search bar.
  }

  if (!event.ctrlKey && !event.altKey && !event.metaKey) {
    switch (event.key) {
      case "f": // freeze all
        node.datum(freeze_node);
        break;
      case "x": // unfreeze all
        node.datum(unfreeze_node);
        simulation.alphaTarget(.3).restart()
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
        // "Display" mode (dark background)
        d3.select("body").attr('id', null).attr('id', "display")
        break;
      case "z":
        // turn off gravity (??)
        simulation
          .force('charge', d3.forceManyBody().strength(0))
          .alphaTarget(.3)
          .restart()
        break
      case "m":
        // spacious layout
        simulation
          .force('charge', d3.forceManyBody().strength(-1200))
          .alphaTarget(.3)
          .restart()
        break
      case "e":
        // tight layout
        simulation
          .force('charge', d3.forceManyBody().strength(-60))
          .alphaTarget(.3)
          .restart()
        break
      default:
        return; // Quit when this doesn't handle the key event.
    }
  event.preventDefault();
  }
  // Cancel the default action to avoid it being handled twice
}, true);



/*
* Save and load config
*
*/

// save config
document.getElementById('download-config').addEventListener('click', function(e){
  let config = {}
  node.data().forEach(node => {
      let config_node = {}

      if ( node.hasOwnProperty('fx') && node.hasOwnProperty('fy') ) {
        config_node['fx'] = config_node['x'] = node['fx']
        config_node['fy'] = config_node['y'] = node['fy']
      } else {
        config_node['x'] = node['x']
        config_node['y'] = node['y']
      }
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
    if ( d.hasOwnProperty('fx') && !load_node_config.hasOwnProperty('fx') ) {
      delete d['fx']
    }
    if ( d.hasOwnProperty('fy') && !load_node_config.hasOwnProperty('fy') ) {
      delete d['fy']
    }
    return d;
  })
  simulation.restart()
}
