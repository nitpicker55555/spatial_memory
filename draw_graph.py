from flask import Flask, render_template
import json
import networkx as nx
from pathlib import Path

app = Flask(__name__)
topic='zork1'
DATA_DIR = Path(rf"D:/mango/data/{topic}")

EDGE_FILE = DATA_DIR / f"{topic}.edges.json"
GRAPH_DATA_FILE = Path("static/graph_data.json")

def build_maze_graph_oneway(edge_file: Path) -> nx.DiGraph:
    with edge_file.open("r", encoding="utf-8") as f:
        edges_data = json.load(f)
    graph = nx.DiGraph()
    for edge in edges_data:
        if edge["seen_in_forward"] != 9999:
            src = edge["src_node"].lower()
            dst = edge["dst_node"].lower()
            action = edge["action"].lower()
            graph.add_edge(src, dst, action=action)
    return graph

def directional_layout_3d(graph: nx.DiGraph, scale=55.0) -> dict:
    direction_vectors = {
        "north": (0, 1, 0),
        "south": (0, -1, 0),
        "east": (1, 0, 0),
        "west": (-1, 0, 0),
        "up": (0, 0, 1),
        "down": (0, 0, -1),
        "northeast": (1, 1, 0),
        "southwest": (-1, -1, 0),
        "go panel": (1, 0.5, 0),
        "follow mouse": (-1, 0.5, 0),
        "pray":(0, -1, 0),
    }

    pos = {}
    visited = set()

    def dfs(node, x, y, z):
        if node in visited:
            return
        visited.add(node)
        pos[node] = (x, y, z)
        for neighbor in graph.successors(node):
            action = graph[node][neighbor]["action"]
            dx, dy, dz = direction_vectors.get(action, (0.3, 0.3, 0.3))
            dfs(neighbor, x + dx * scale, y + dy * scale, z + dz * scale)

    if len(graph.nodes) == 0:
        return {}
    start = next(iter(graph.nodes))
    dfs(start, 0, 0, 0)
    return pos


def export_graph_json():
    G = build_maze_graph_oneway(EDGE_FILE)
    pos = directional_layout_3d(G)

    base_directions = {"north", "south", "east", "west", "up", "down"}
    color_map = {
        "up": "gray",
        "down": "gray",
        "north": "gray",
        "south": "gray",
        "east": "gray",
        "west": "gray",
    }

    reverse_direction = {
        "north": "south", "south": "north",
        "east": "west", "west": "east",
        "up": "down", "down": "up",
        "northeast": "southwest", "southwest": "northeast",
        "northwest": "southeast", "southeast": "northwest"
    }

    nodes = [
        {"id": n, "x": x, "y": y, "z": z, "fx": x, "fy": y, "fz": z}
        for n, (x, y, z) in pos.items()
    ]

    links = []

    for u, v, data in G.edges(data=True):
        action = data["action"]
        links.append({
            "source": u,
            "target": v,
            "action": action,
            "color": color_map.get(action, "gray"),
            "width": 1
        })

        # Check and add reverse edge if missing and direction is standard
        if action in base_directions and not G.has_edge(v, u):
            reverse_action = reverse_direction.get(action)
            links.append({
                "source": v,
                "target": u,
                "action": reverse_action + " (auto)",
                "color": "red",
                "width": 2
            })

    data = {"nodes": nodes, "links": links}
    GRAPH_DATA_FILE.parent.mkdir(exist_ok=True)
    with GRAPH_DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

@app.route("/")
def index():
    export_graph_json()
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
