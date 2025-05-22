import json
import os
import os.path as osp  # os.path is more conventional than os.osp
import networkx as nx
from tqdm import tqdm

from eval_rf import validate_rf_answer
from problem_extraction import analyze_and_extract_problems
from load_json_graph import load_graph_from_json
def _get_game_info(map_dir: str, game_name: str):
    """
    Loads game information from files.
    (Adapted from mango/evaluation/utils/map_utils.py)
    """
    all2all_path = osp.join(map_dir, game_name, f"{game_name}.all2all.jsonl")
    all_pairs_path = osp.join(map_dir, game_name,
                              f"{game_name}.all_pairs.jsonl")
    edges_path = osp.join(map_dir, game_name, f"{game_name}.edges.json")
    actions_path = osp.join(map_dir, game_name, f"{game_name}.actions.json")
    locations_path = osp.join(map_dir, game_name,
                              f"{game_name}.locations.json")
    walkthrough_path = osp.join(map_dir, game_name, f"{game_name}.walkthrough")

    data_files = {
        "all2all": all2all_path, "all_pairs": all_pairs_path,
        "edges": edges_path,
        "actions": actions_path, "locations": locations_path,
        "walkthrough": walkthrough_path
    }

    for key, path in data_files.items():
        if not osp.exists(path):
            raise FileNotFoundError(
                f"Required data file not found: {path} for game '{game_name}' in map_dir '{map_dir}'")

    all2all = {}
    with open(all2all_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            all2all[data["id"]] = data

    all_pairs = {}
    with open(all_pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            all_pairs[data["id"]] = data

    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    with open(actions_path, "r", encoding="utf-8") as f:
        actions = json.load(f)

    with open(locations_path, "r", encoding="utf-8") as f:
        locations = json.load(f)

    with open(walkthrough_path, "r", encoding="utf-8") as f:
        walkthrough = f.read()

    G = nx.MultiDiGraph()
    for edge in edges:
        G.add_edge(
            edge["src_node"],
            edge["dst_node"],
            action=edge["action"],
            # Assuming these keys exist based on original file, add error handling if necessary
            edge_min_step_answerable=edge.get("edge_min_step_answerable"),
            seen_in_forward_answerable=edge.get("seen_in_forward_answerable"),
        )

    # We are primarily interested in the graph G for this function's purpose
    # but returning others for consistency with original util if it were to be reused.
    return G, all2all, all_pairs, actions, locations, walkthrough


def _get_game_info_with_G_eval(map_dir: str, game_name: str):
    """
    Gets game info and constructs G_eval with reverse edges.
    (Adapted from mango/evaluation/utils/map_utils.py)
    """
    reverse_dict = {
        "up": "down", "down": "up",
        "north": "south", "south": "north",
        "east": "west", "west": "east",
        "northeast": "southwest", "northwest": "southeast",
        "southeast": "northwest", "southwest": "northeast",
    }

    G, all2all, all_pairs, actions, locations, walkthrough = _get_game_info(
        map_dir, game_name
    )

    G_eval = nx.MultiDiGraph()
    for u, v, data in G.edges(data=True):
        G_eval.add_edge(u, v, **data)  # Copy original edges

    # Add reverse edges
    for u, v, data in G.edges(data=True):
        action_ = data.get("action")
        if action_ and action_.lower() in reverse_dict:  # Ensure action exists and is in dict
            reverse_action = reverse_dict[
                action_.lower()]  # Use lower for robustness

            # Check if reverse edge with this specific reverse_action already exists
            edge_exists = False
            if G_eval.has_edge(v, u):
                for _, _, existing_data in G_eval.edges(v, data=True):
                    if G_eval.nodes[u] == _ and existing_data.get("action",
                                                                  "").lower() == reverse_action:  # Check specific target and action
                        edge_exists = True
                        break

            if not edge_exists:
                # Create a new data dictionary for the reverse edge
                reverse_data = data.copy()  # Start with a copy of original data
                reverse_data["action"] = reverse_action
                # Potentially adjust or remove other attributes like 'edge_min_step_answerable'
                # if they don't make sense for an auto-generated reverse edge.
                # For now, keeping them for simplicity.
                G_eval.add_edge(v, u, **reverse_data)

    return G_eval, G, actions, locations, all2all, all_pairs, walkthrough


def load_correct_graph_from_dataset(
        game_dataset_folder_path: str) -> nx.MultiDiGraph:
    """
    Loads the 'correct' evaluation graph (G_eval) from a game dataset folder.

    The game_dataset_folder_path should be the direct path to a specific game's
    data files (e.g., './data/905/').

    Args:
        game_dataset_folder_path (str): Path to the game's dataset folder.

    Returns:
        networkx.MultiDiGraph: The G_eval graph for the dataset, including reverse edges.

    Raises:
        FileNotFoundError: If required data files (e.g., .edges.json) are not found.
        Exception: For other potential errors during loading or graph construction.
    """
    if not osp.isdir(game_dataset_folder_path):
        raise FileNotFoundError(
            f"The provided game dataset folder path does not exist or is not a directory: {game_dataset_folder_path}")

    map_dir = osp.dirname(game_dataset_folder_path)
    game_name = osp.basename(game_dataset_folder_path)

    if not game_name:  # Handles cases like "data/" vs "data/game_name"
        print(
            "Could not determine game_name from path. Please provide path to specific game folder.")
    if not map_dir and game_name == game_dataset_folder_path:  # e.g. if path is just "905"
        map_dir = "."  # Assume current directory if only game_name is given as path

    try:
        # We only need G_eval from the returned tuple
        g_eval, _, _, _, _, _, _ = _get_game_info_with_G_eval(map_dir,
                                                              game_name)
        return g_eval
    except FileNotFoundError as e:
        print(f"Error loading graph data for {game_name} from {map_dir}: {e}")
        raise
    except Exception as e:
        print(
            f"An unexpected error occurred while generating the graph for {game_name}: {e}")
        raise


def get_path_edge_sequence(graph: nx.MultiDiGraph, key_problem:dict
                           ):
    """
    Returns the path from start to end as a list of edge descriptions:
    [{"prev_loc": ..., "command": ..., "next_loc": ...}, ...]

    Assumes edges contain an 'action' attribute.
    If multiple edges exist between nodes, the first valid one is used.
    """
    start=key_problem['src_node']
    end=key_problem['dst_node']
    if not graph.has_node(start):
        print(f"Start node '{start}' not in graph.")
        return f"Start node '{start}' not in graph."

    if not graph.has_node(end):

        print(f"End node '{end}' not in graph.")
        return f"End node '{end}' not in graph."


    try:
        # Use shortest path based on nodes (not edge weights)
        path_nodes = nx.shortest_path(graph, source=start, target=end)
    except nx.NetworkXNoPath:
        print(f"No path found from '{start}' to '{end}'.")
        return f"No path found from '{start}' to '{end}'."

    edge_sequence = []

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]

        # Choose the first edge with an action (you can customize this logic)
        found = False
        for _, edge_data in graph.get_edge_data(u, v).items():
            action = edge_data.get("action")
            if action:
                edge_sequence.append({
                    "prev_loc": u,
                    "command": action,
                    "next_loc": v
                })
                found = True
                break
        if not found:
            print(
                f"No valid action edge found between '{u}' and '{v}'.")
            return f"No valid action edge found between '{u}' and '{v}'."
    return edge_sequence
def validate_df_answer(G, path_json: dict) -> bool:
    """
    Verifies whether a given path from src_node to dst_node is valid in the graph defined in graph_json_path.

    Args:
        graph_json_path (str): Path to the graph JSON file.
        path_json (dict): A dictionary containing 'src_node', 'dst_node', and 'path_details'.

    Returns:
        bool: True if the path is valid and ends at the expected dst_node, False otherwise.
    """
    # if not osp.isfile(graph_json_path):
    #     raise FileNotFoundError(f"Graph file not found: {graph_json_path}")
    #
    # # Load the graph
    # with open(graph_json_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    # G = nx.MultiDiGraph()

    # for node in data.get("nodes", []):
    #     G.add_node(node["id"])
    #
    # for link in data.get("links", []):
    #     source = link["source"]
    #     target = link["target"]
    #     action = link.get("action", "")
    #     G.add_edge(source, target, action=action)

    # Traverse path
    current_node = path_json["src_node"]
    for step in path_json["path_details"]:
        expected_prev = step["prev_node"]
        next_node = step["node"]
        action = step["action"]

        if expected_prev != current_node:
            print(f"Path mismatch: expected previous node {expected_prev}, but current is {current_node}")
            return f"Path mismatch: expected previous node {expected_prev}, but current is {current_node}"

        found = False
        for _, tgt, edge_data in G.out_edges(current_node, data=True):
            if tgt == next_node and edge_data.get("action") == action:
                found = True
                break

        if not found:
            print(f"No edge from '{current_node}' to '{next_node}' with action '{action}'")
            return f"No edge from '{current_node}' to '{next_node}' with action '{action}'"

        current_node = next_node

    if current_node != path_json["dst_node"]:
        print(f"Final node mismatch: expected '{path_json['dst_node']}', got '{current_node}'")
        return f"Final node mismatch: expected '{path_json['dst_node']}', got '{current_node}'"

    return True


# --- Example Usage ---
if __name__ == "__main__":
    dummy_map_dir = r"data\data"
    output_file = "validation_generated_results_rf.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        pass  # 或 f.write("")
    results = []
    valid_list=['905', 'afflicted', 'anchor', 'ballyhoo', 'cutthroat', 'dragon',
     'enchanter', 'enter', 'gold', 'hhgg', 'hollywood', 'huntdark', 'infidel',
     'jewel', 'library', 'loose', 'lostpig', 'ludicorp', 'lurking',
     'partyfoul', 'pentari', 'planetfall', 'plundered', 'sherlock',
     'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'trinity',
     'yomomma', 'zenon', 'ztuu']

    for dummy_game_name in os.listdir(dummy_map_dir):
        if dummy_game_name in valid_list:
            dummy_game_folder = osp.join(dummy_map_dir, dummy_game_name)
            if not osp.isfile(osp.join(dummy_map_dir, dummy_game_name,f'{dummy_game_name}.all2all.jsonl')):
                print(osp.join(dummy_map_dir, dummy_game_name,f'{dummy_game_name}.all2all.jsonl'))
                continue

            print(f"Attempting to load graph from: {dummy_game_folder}")
            correct_graph = load_graph_from_json(dummy_game_name)
            # correct_graph = load_correct_graph_from_dataset(dummy_game_folder)

            if not correct_graph:
                print(f"Failed to load graph for game: {dummy_game_name}")
                continue

            print(f"\nSuccessfully loaded graph for game: {dummy_game_name}")
            print(f"Number of nodes: {correct_graph.number_of_nodes()}")

            unique_edges = set(
                (u, v, data.get("action"))
                for u, v, data in correct_graph.edges(data=True)
            )
            print("Edges (with data):", len(unique_edges))

            for u, v, action in sorted(unique_edges):
                print(f"  {u} -> {v} (action: {action})")

            # key_problem = {
            #     "src_node": "front of house",
            #     "dst_node": "dark",
            # }
            analyze_result=analyze_and_extract_problems(dummy_game_name,70)
            difficulties= ['easy_problems','hard_problems']
            for difficulty in difficulties:
                for each_rf_question in tqdm(analyze_result['route_finding'][difficulty]):
                    edge_answer = get_path_edge_sequence(correct_graph, each_rf_question)
                    if isinstance(edge_answer,str):
                        validation_result= {"full_path_is_correct":False,'reason':edge_answer}
                    else:
                        validation_result = validate_rf_answer(each_rf_question, edge_answer,
                                                           correct_graph)
                    result_record = {
                        "game": dummy_game_name,
                        "key_problem": each_rf_question,
                        "edge_answer": edge_answer,
                        "validation_result": validation_result,
                        "diffculty": difficulty
                    }
            #     for each_df_question in tqdm(analyze_result['desti_finding'][difficulty]):
            #         # edge_answer = get_path_edge_sequence(correct_graph, each_rf_question)
            #         # if isinstance(edge_answer,str):
            #         #     validation_result= {"full_path_is_correct":False,'reason':edge_answer}
            #         # else:
            #         validation_result = validate_df_answer(correct_graph,each_df_question)
            #         if isinstance(validation_result,str):
            #             validation_result_json={'validation_result':False,'reason':validation_result}
            #         else:
            #             validation_result_json={'validation_result':True}
            #
            #         # print("validation_result",validation_result['full_path_is_correct'])
            #         result_record = {
            #             "game": dummy_game_name,
            #             "key_problem": each_df_question,
            #             # "edge_answer": edge_answer,
            #             "validation_result": validation_result_json,
            #             "diffculty":difficulty
            #         }
                    results.append(result_record)

                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    print("All validations complete. Results saved to:", output_file)

