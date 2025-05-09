fetch("/graph")
  .then(res => res.json())
  .then(data => {
    const directionalVectors = {
      up: [0, 1, 0],
      down: [0, -1, 0],
      north: [0, 0, 1],
      south: [0, 0, -1],
      east: [1, 0, 0],
      west: [-1, 0, 0],
      northeast: [1, 1, 0],
      southwest: [-1, -1, 0],
      "go panel": [1, 0.5, 0],
      "follow mouse": [-1, 0.5, 0],
    };

    const positions = {};
    positions[data.nodes[0].id] = { x: 0, y: 0, z: 0 };

    data.links.forEach(link => {
      const { source, target, action } = link;
      const base = positions[source] || { x: 0, y: 0, z: 0 };
      const [dx, dy, dz] = directionalVectors[action] || [0.3, 0.3, 0.3];
      positions[target] = { x: base.x + dx, y: base.y + dy, z: base.z + dz };
    });

const Graph = ForceGraph3D()(document.getElementById("3d-graph"))
  .graphData(data)
  .nodeLabel("id")
  .linkLabel(link => link.action)
  .nodeThreeObject(node => {
    const sprite = new SpriteText(node.id);
    sprite.color = "lightblue";
    sprite.textHeight = 1.5;
    return sprite;
  })
  .nodePositionUpdate(node => {
    const pos = positions[node.id];
    if (pos) {
      node.x = pos.x;
      node.y = pos.y;
      node.z = pos.z;
    }
  })
  .d3Force("charge", null)         // 禁用斥力
  .d3VelocityDecay(1);             // 禁用自动漂移
