from flask import Flask, jsonify, render_template, stream_with_context, \
    Response, request
from flask_cors import CORS
# from extract_grid import *
app = Flask(__name__)
CORS(app)
from chat_py import chat_single, message_template
from eval import read_walkthrough
from pathlib import Path
DATA_DIR = Path(r"D:/mango/data/night")


walk_data=walkthrough=read_walkthrough(DATA_DIR / "night.walkthrough",70)



node_sequence = [
    {'name': 'Computer Site', 'coordinates': (0, 0, 0)},
    {'name': 'Hall Outside Computer Site', 'coordinates': (1, 1, 0)},
    {'name': 'Hall', 'coordinates': (1, 0, 0)},
    {'name': 'Hall Outside Elevator', 'coordinates': (2, 0, 0)},
    {'name': 'Stairwell (Third Floor)', 'coordinates': (0, 1, 0)},
    {'name': 'Stairwell (Second Floor)', 'coordinates': (0, 1, -1)},
    {'name': 'Stairwell (First Floor)', 'coordinates': (0, 1, -2)},
    {'name': 'Hall (First Floor)', 'coordinates': (1, 1, -2)},
    {'name': 'Hall (Middle)', 'coordinates': (1, 0, -2)},
    {'name': 'Hall Outside Elevator (First Floor)', 'coordinates': (2, 0, -2)},
    {'name': "Janitor's Closet", 'coordinates': (2, -1, -2)},
    {'name': 'Outside Physics Office', 'coordinates': (1, 1, -1)},
    {'name': 'Maze of Twisty Passages', 'coordinates': (1, 0, -3)},
    {'name': "Gnome's Lair", 'coordinates': (0, 0, -3)},
]
def scale_coordinates(node_sequence, scale=50):
    scaled_nodes = []
    for node in node_sequence:
        name = node['name']
        x, y, z = node['coordinates']
        scaled_coords = (x * scale, y * scale, z * scale)
        scaled_nodes.append({'name': name, 'coordinates': scaled_coords})
    return scaled_nodes
node_sequence=scale_coordinates(node_sequence)
links = [

    {"source": "Computer Site", "target": "Hall Outside Computer Site",
     "action": "northeast"},
    {"source": "Hall Outside Computer Site", "target": "Hall",
     "action": "south"},
    {"source": "Hall", "target": "Hall Outside Elevator", "action": "east"},
    {"source": "Hall Outside Elevator", "target": "Stairwell (Third Floor)",
     "action": "northwest"},
    {"source": "Stairwell (Third Floor)", "target": "Stairwell (Second Floor)",
     "action": "down"},
    {"source": "Stairwell (Second Floor)", "target": "Stairwell (First Floor)",
     "action": "down"},
    {"source": "Stairwell (First Floor)", "target": "Hall (First Floor)",
     "action": "east"},
    {"source": "Hall (First Floor)", "target": "Hall (Middle)",
     "action": "south"},
    {"source": "Hall (Middle)",
     "target": "Hall Outside Elevator (First Floor)", "action": "east"},
    {"source": "Hall Outside Elevator (First Floor)",
     "target": "Janitor's Closet", "action": "south"},
    {"source": "Janitor's Closet", "target": "Outside Physics Office",
     "action": "northwest"},
    {"source": "Outside Physics Office", "target": "Maze of Twisty Passages",
     "action": "south"},
    {"source": "Maze of Twisty Passages", "target": "Gnome's Lair",
     "action": "west"},
]



current_index = 0


@app.route("/")
def index():
    return render_template("real_time.html")
def get_graph_state(n):


    nodes = node_sequence[:n]
    node_ids = {node["name"] for node in nodes}
    filtered_links = [link for link in links if link["source"] in node_ids and link["target"] in node_ids]

    graph_nodes = [
        {"id": node["name"], "x": node["coordinates"][0], "y": node["coordinates"][1], "z": node["coordinates"][2]}
        for node in nodes
    ]

    return {
        "nodes": graph_nodes,
        "links": filtered_links,
        "done": n >= len(node_sequence) - 1
    }


@app.route("/graph_data")
def graph_data():
    global current_index

    if request.args.get("reset") == "true":
        current_index = 0

    graph_state = get_graph_state(current_index + 1)

    if not graph_state["done"]:
        current_index += 1

    return jsonify(graph_state)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9090)
