import json
import networkx as nx
import os.path as osp

def load_graph_from_json(game_name) -> nx.MultiDiGraph:
    """
    Loads a graph from a JSON file in the specified format and returns it as a MultiDiGraph.

    Args:
        json_file_path (str): Path to the JSON file containing the graph.

    Returns:
        networkx.MultiDiGraph: The constructed graph with reverse edges included.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        Exception: For other errors during file reading or graph construction.
    """
    json_file_path=fr"D:\spatial_memory\history_graph\llm_maze_graph_{game_name}.json"
    if not osp.isfile(json_file_path):
        raise FileNotFoundError(f"The specified JSON file was not found: {json_file_path}")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        G = nx.MultiDiGraph()

        # Add nodes
        for node in data.get("nodes", []):
            G.add_node(node["id"])

        # Add edges and reverse edges
        for link in data.get("links", []):
            source = link["source"]
            target = link["target"]
            action = link.get("action", "")

            # Original edge
            G.add_edge(source, target, action=action)

            # Reverse edge
            reverse_action = f"reverse_{action}" if action else "reverse"
            G.add_edge(target, source, action=reverse_action)

        return G

    except Exception as e:
        print(f"Error loading graph from {json_file_path}: {e}")
        raise
# load_graph_from_json('curses')