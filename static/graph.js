function drawRootNodes(nodes, links) {
  // Specify the dimensions of the chart.
  const width = 928;
  const height = 680;

  // Create a simulation with several forces.
  const simulation = d3.forceSimulation(nodes)
      .force("charge", d3.forceManyBody())
      .force("x", d3.forceX())
      .force("y", d3.forceY());

  // Create the SVG container.
  const svg = d3.create("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [-width / 2, -height / 2, width, height])
      .attr("style", "max-width: 100%; height: auto;");

  const node = svg.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
      .attr("r", 5)
      .attr("fill", "green")


  // Add a line for each link, and a circle for each node.
  const link = svg.append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(links)
    .join("line")
      .attr("stroke-width", 1);

  node.append("title")
      .text(d => d.id);

  node.call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));
  
  simulation.on("tick", () => {
    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
  });

  // Reheat the simulation when drag starts, and fix the subject position.
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  // Update the subject (dragged node) position during drag.
  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  // Restore the target alpha so the simulation cools after dragging ends.
  // Unfix the subject position now that it’s no longer being dragged.
  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return svg.node();
}

document.addEventListener('DOMContentLoaded', (event) => {
	fetch('/nodes')
		.then(response => response.json())
		.then(data => {
      console.log(data);
      const graphSVG = drawRootNodes(data, []);
      document
        .getElementById("graph")
        .appendChild(graphSVG);
		});
});
