import json
import os
import os.path as osp
import networkx as nx
from collections import deque
from enum import Enum  # Using Enum for better readability for EvalMetric


# --- Prerequisite: Graph Loading Function (from previous response) ---
# This function is needed if you want to load the graph within the validation flow.
# Or, you can pass the graph as an argument if already loaded.

def _get_game_info_for_graph(map_dir: str, game_name: str):
    edges_path = osp.join(map_dir, game_name, f"{game_name}.edges.json")
    if not osp.exists(edges_path):
        raise FileNotFoundError(f"Required data file not found: {edges_path}")

    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    G = nx.MultiDiGraph()
    for edge in edges:
        G.add_edge(
            edge["src_node"],
            edge["dst_node"],
            action=edge["action"],
            edge_min_step_answerable=edge.get("edge_min_step_answerable"),
            seen_in_forward_answerable=edge.get("seen_in_forward_answerable"),
        )
    return G


def load_correct_graph_from_dataset_for_validation(
        game_dataset_folder_path: str) -> nx.MultiDiGraph:
    if not osp.isdir(game_dataset_folder_path):
        raise FileNotFoundError(
            f"Game dataset folder path does not exist: {game_dataset_folder_path}")

    map_dir = osp.dirname(game_dataset_folder_path)
    game_name = osp.basename(game_dataset_folder_path)
    if not game_name:
        raise ValueError("Could not determine game_name from path.")
    if not map_dir and game_name == game_dataset_folder_path:
        map_dir = "."

    reverse_dict = {
        "up": "down", "down": "up", "north": "south", "south": "north",
        "east": "west", "west": "east", "northeast": "southwest",
        "northwest": "southeast", "southeast": "northwest",
        "southwest": "northeast",
    }

    G = _get_game_info_for_graph(map_dir, game_name)
    G_eval = nx.MultiDiGraph()
    for u, v, data in G.edges(data=True):
        G_eval.add_edge(u, v, **data)

    for u, v, data in list(
            G.edges(data=True)):  # Use list to avoid issues if G changes, though G_eval is being modified
        action_ = data.get("action")
        if action_ and action_.lower() in reverse_dict:
            reverse_action = reverse_dict[action_.lower()]
            edge_exists = False
            if G_eval.has_edge(v, u):
                for _, _, existing_data_bundle in G_eval.edges(v, keys=False,
                                                               data=True):
                    if existing_data_bundle.get("action",
                                                "").lower() == reverse_action:
                        edge_exists = True
                        break
            if not edge_exists:
                reverse_data = data.copy()
                reverse_data["action"] = reverse_action
                G_eval.add_edge(v, u, **reverse_data)
    return G_eval


# --- Validation Logic Components ---

class EvalMetric(Enum):
    STRICT = 0
    LOOSE = 1


def edit_distance_score(s1: str, s2: str) -> float:
    """Normalized edit distance score (1 is perfect match)."""
    s1 = s1.lower()
    s2 = s2.lower()
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    max_len = max(m, n)
    return (1 - dp[m][n] / max_len) if max_len != 0 else 1.0 if m == n else 0.0


def _matching_score_rf(s1: str, s2: str, metric: EvalMetric) -> float:
    """Calculates matching score based on metric."""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    if metric == EvalMetric.STRICT:
        return 1.0 if s1_lower == s2_lower else 0.0
    elif metric == EvalMetric.LOOSE:
        return edit_distance_score(s1_lower, s2_lower)
    return 0.0


def _bfs_get_multi_des_rf(graph: nx.MultiDiGraph, src_node: str,
                          actions_sequence: list[str],
                          action_attribute_key="action") -> list[str]:
    """
    Performs BFS based on a sequence of actions to find possible destination nodes.
    Actions in actions_sequence are assumed to be canonical game actions.
    """
    dest_nodes = set()  # Use set to store unique destination nodes
    if not graph.has_node(src_node):
        return []

    queue = deque([(src_node, 0)])  # (current_node, action_step_index)

    visited_states = set()  # To handle cycles with (node, step_index)

    while queue:
        current_node, step_idx = queue.popleft()

        if (current_node, step_idx) in visited_states:
            continue
        visited_states.add((current_node, step_idx))

        if step_idx == len(actions_sequence):
            dest_nodes.add(current_node)
            continue  # Path ended, collect destination

        if step_idx < len(actions_sequence):
            target_action = actions_sequence[step_idx].lower()
            for _, next_node, edge_data in graph.out_edges(current_node,
                                                           data=True):
                edge_action = edge_data.get(action_attribute_key)
                if edge_action and edge_action.lower() == target_action:
                    if (next_node,
                        step_idx + 1) not in visited_states:  # Optimization
                        queue.append((next_node, step_idx + 1))
    return list(dest_nodes)


def _evaluate_actions_logic_rf(
        graph: nx.MultiDiGraph,
        parsed_llm_path: list[dict],  # LLM's proposed path, list of dicts
        src_node: str,
        dst_node: str,
        metric: EvalMetric,
        llm_action_key="action",  # Key for action in llm_parsed_path items
        graph_action_key="action"  # Key for action in graph edge data
) -> float:
    """
    Evaluates if the sequence of actions in the LLM's path leads to the destination.
    """
    # 1. Get all canonical actions from the graph
    actions_gt = list(set(
        d[graph_action_key].lower() for _, _, d in graph.edges(data=True) if
        graph_action_key in d))
    if not actions_gt:  # No actions in graph, cannot proceed
        return 0.0

    # 2. Extract actions from LLM's path and map them to canonical actions
    llm_actions_raw = []
    for step in parsed_llm_path:
        action_val = step.get(llm_action_key)
        if isinstance(action_val, str):
            llm_actions_raw.append(action_val)
        # else: could be an error or different format, for now just skipping

    mapped_llm_actions = []
    for llm_action_str in llm_actions_raw:
        if not llm_action_str.strip(): continue  # Skip empty actions
        # Find the best matching canonical action
        best_match_action = max(
            actions_gt,
            key=lambda canonical_act: _matching_score_rf(llm_action_str,
                                                         canonical_act,
                                                         EvalMetric.LOOSE)
            # Always use loose for mapping
        )
        # Optional: Add a threshold for matching quality here if needed
        mapped_llm_actions.append(best_match_action)

    if not mapped_llm_actions and llm_actions_raw:  # If actions were present but none could be mapped (e.g. actions_gt was empty or no good match)
        return 0.0
    if not mapped_llm_actions and not llm_actions_raw:  # No actions provided by LLM
        # If src == dst and no actions, it's correct. Otherwise, incorrect.
        return 1.0 if src_node == dst_node else 0.0

    # 3. Simulate path using BFS
    reached_destinations = _bfs_get_multi_des_rf(graph, src_node,
                                                 mapped_llm_actions,
                                                 graph_action_key)

    # 4. Score based on whether dst_node is among reached_destinations
    if not reached_destinations:
        return 0.0

    final_score = 0.0
    for reached_node in reached_destinations:
        final_score = max(final_score,
                          _matching_score_rf(dst_node, reached_node, metric))
    return final_score


def _evaluate_full_path_logic_rf(
        graph: nx.MultiDiGraph,
        parsed_llm_path: list[dict],  # LLM's proposed path
        src_node: str,
        dst_node: str,
        loc_before_key="location_before",
        action_key="action",
        loc_after_key="location_after",
        graph_action_key="action"  # Key for action in graph edge data
) -> bool:
    """
    Evaluates the structural integrity and step-by-step validity of the LLM's path.
    Node names from LLM path and src/dst nodes are compared case-insensitively
    by converting them to lowercase. This assumes graph nodes are also stored/accessible
    in lowercase. For consistent behavior, ensure graph nodes are stored in lowercase.
    """
    # Normalize src_node and dst_node from function arguments for consistent comparison
    src_node_norm = src_node.lower()
    dst_node_norm = dst_node.lower()

    if not parsed_llm_path:
        return src_node_norm == dst_node_norm  # Empty path is correct only if src is already dst

    # Check start and end nodes of the entire path
    # Normalize node names from the LLM path for comparison
    first_step_loc_before = parsed_llm_path[0].get(loc_before_key,
                                                   "").strip().lower()
    if first_step_loc_before != src_node_norm:
        # print(f"DEBUG: Path start mismatch: LLM '{first_step_loc_before}' vs Expected '{src_node_norm}'")
        return False

    last_step_loc_after = parsed_llm_path[-1].get(loc_after_key,
                                                  "").strip().lower()
    if last_step_loc_after != dst_node_norm:
        # print(f"DEBUG: Path end mismatch: LLM '{last_step_loc_after}' vs Expected '{dst_node_norm}'")
        return False

    current_step_data_for_error = {}  # For debugging output
    current_step_index_for_error = -1  # For debugging output

    try:
        for i, step in enumerate(parsed_llm_path):
            current_step_data_for_error = step
            current_step_index_for_error = i

            # Normalize all parts of the current step from LLM data
            loc_before_llm = step.get(loc_before_key, "").strip().lower()
            action_llm = step.get(action_key, "").strip().lower()
            loc_after_llm = step.get(loc_after_key, "").strip().lower()

            if not loc_before_llm or not action_llm or not loc_after_llm:
                # print(f"DEBUG: Malformed step {i}: {step}")
                return False  # Malformed step, essential parts are missing

            # Validate the current step against the graph
            # This assumes loc_before_llm and loc_after_llm (lowercase versions)
            # are used as node identifiers if graph nodes are consistently lowercased.
            step_is_valid_in_graph = False
            if graph.has_node(loc_before_llm) and graph.has_node(
                    loc_after_llm):
                # Check if any edge exists between these nodes first
                if graph.has_edge(loc_before_llm, loc_after_llm):
                    # Iterate through all parallel edges (for MultiDiGraph)
                    for edge_key in graph[loc_before_llm][loc_after_llm]:
                        edge_data = graph[loc_before_llm][loc_after_llm][
                            edge_key]
                        graph_edge_action = edge_data.get(graph_action_key,
                                                          "").strip().lower()

                        if graph_edge_action == action_llm:
                            step_is_valid_in_graph = True
                            break  # Found a matching edge for this step

            if not step_is_valid_in_graph:
                # print(f"DEBUG: Step {i} not valid in graph: {loc_before_llm} -({action_llm})-> {loc_after_llm}")
                return False

            # Check path continuity: current step's loc_after must match next step's loc_before
            if i < len(parsed_llm_path) - 1:
                next_step_loc_before_llm = parsed_llm_path[i + 1].get(
                    loc_before_key, "").strip().lower()
                if loc_after_llm != next_step_loc_before_llm:
                    # print(f"DEBUG: Path continuity error at step {i}: '{loc_after_llm}' != '{next_step_loc_before_llm}'")
                    return False
    except Exception as e:
        # This block catches any other unexpected errors during processing.
        print(
            f"An unexpected error occurred during path evaluation at step {current_step_index_for_error}:")
        print(
            f"Problematic LLM Path (around error, if available): {parsed_llm_path[max(0, current_step_index_for_error - 1): current_step_index_for_error + 2]}")
        print(
            f"Current step data being processed: {current_step_data_for_error}")
        print(f"Error details: {type(e).__name__}: {e}")
        # For a validation function, returning False on error is often preferred.
        return False

    return True


# --- Main Validation Function ---
def validate_rf_answer(
        rf_problem: dict,  # e.g., {"src_node": "A", "dst_node": "B"}
        llm_parsed_answer: list[dict],
        # Parsed path, e.g., [{"location_before":"A", "action":"go N", "location_after":"B"}]
        game_graph: nx.MultiDiGraph,  # The G_eval graph for the specific game
        path_keys: dict = None
        # Optional: {"loc_before":"prev", "action":"act", "loc_after":"next"}
) -> dict:
    """
    Validates a Route Finding (RF) answer against the game graph.

    Args:
        rf_problem (dict): Dictionary with "src_node" and "dst_node".
        llm_parsed_answer (list[dict]): The LLM's proposed path, parsed into a list of step dictionaries.
        game_graph (nx.MultiDiGraph): The evaluation graph (G_eval) for the game.
        path_keys (dict, optional): Optional mapping for keys in llm_parsed_answer steps.
                                    Defaults to {"loc_before":"location_before", "action":"action", "loc_after":"location_after"}.

    Returns:
        dict: A dictionary containing validation results:
              {
                  "strict_action_score": float (0.0 or 1.0),
                  "loose_action_score": float (0.0 to 1.0),
                  "full_path_is_correct": bool,
                  "src_node": str,
                  "dst_node": str,
                  "llm_path_steps": int
              }
    """
    path_keys= {"loc_before": "prev_loc", "action": "command",
                       "loc_after": "next_loc"}
    if not isinstance(rf_problem,
                      dict) or "src_node" not in rf_problem or "dst_node" not in rf_problem:
        raise ValueError(
            "rf_problem must be a dict with 'src_node' and 'dst_node'.")
    if not isinstance(llm_parsed_answer, list):
        # Allow empty list for cases where LLM gives no path
        pass  # raise ValueError("llm_parsed_answer must be a list of path step dictionaries.")
    if not isinstance(game_graph, nx.MultiDiGraph):
        raise ValueError("game_graph must be a networkx.MultiDiGraph.")

    src_node = rf_problem["src_node"]
    dst_node = rf_problem["dst_node"]

    default_keys = {"loc_before": "location_before", "action": "action",
                    "loc_after": "location_after"}
    if path_keys:
        current_keys = {**default_keys, **path_keys}
    else:
        current_keys = default_keys

    loc_b_key = current_keys["loc_before"]
    act_key = current_keys["action"]
    loc_a_key = current_keys["loc_after"]

    # Ensure nodes exist in graph for robust checking, though eval functions might also check
    if not game_graph.has_node(src_node):
        # print(f"Warning: Source node '{src_node}' not in graph.")
        # Depending on desired strictness, could return error or scores of 0
        pass
    if not game_graph.has_node(dst_node):
        # print(f"Warning: Destination node '{dst_node}' not in graph.")
        pass

    strict_action_score = _evaluate_actions_logic_rf(
        graph=game_graph,
        parsed_llm_path=llm_parsed_answer,
        src_node=src_node,
        dst_node=dst_node,
        metric=EvalMetric.STRICT,
        llm_action_key=act_key
    )

    loose_action_score = _evaluate_actions_logic_rf(
        graph=game_graph,
        parsed_llm_path=llm_parsed_answer,
        src_node=src_node,
        dst_node=dst_node,
        metric=EvalMetric.LOOSE,
        llm_action_key=act_key
    )

    full_path_correct = _evaluate_full_path_logic_rf(
        graph=game_graph,
        parsed_llm_path=llm_parsed_answer,
        src_node=src_node,
        dst_node=dst_node,
        loc_before_key=loc_b_key,
        action_key=act_key,
        loc_after_key=loc_a_key
    )

    return {
        "strict_action_score": strict_action_score,
        "loose_action_score": loose_action_score,
        "full_path_is_correct": full_path_correct,
        "src_node": src_node,
        "dst_node": dst_node,
        "llm_path_steps": len(llm_parsed_answer)
    }


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create/Load a dummy graph (G_eval)
    # For this example, let's define a simple one manually
    game_map_graph = nx.MultiDiGraph()
    game_map_graph.add_node("Entrance Hall")
    game_map_graph.add_node("Dining Room")
    game_map_graph.add_node("Kitchen")
    game_map_graph.add_node("Secret Passage")

    game_map_graph.add_edge("Entrance Hall", "Dining Room", action="go north")
    game_map_graph.add_edge("Dining Room", "Entrance Hall",
                            action="go south")  # Reverse for G_eval
    game_map_graph.add_edge("Dining Room", "Kitchen", action="go east")
    game_map_graph.add_edge("Kitchen", "Dining Room", action="go west")
    game_map_graph.add_edge("Dining Room", "Secret Passage",
                            action="pull lever")
    game_map_graph.add_edge("Secret Passage", "Kitchen",
                            action="crawl through tunnel")
    # Add a slightly misspelled action to test loose matching
    game_map_graph.add_edge("Kitchen", "Pantry", action="open door")
    game_map_graph.add_edge("Pantry", "Kitchen", action="close door")

    # 2. Define an RF problem
    problem1 = {"src_node": "Entrance Hall", "dst_node": "Kitchen"}

    # 3. Define some LLM answers (parsed paths)
    # Answer 1: Correct and optimal
    answer1_path = [
        {"location_before": "Entrance Hall", "action": "go north",
         "location_after": "Dining Room"},
        {"location_before": "Dining Room", "action": "go east",
         "location_after": "Kitchen"}
    ]

    # Answer 2: Correct but suboptimal (via secret passage)
    answer2_path = [
        {"location_before": "Entrance Hall", "action": "go north",
         "location_after": "Dining Room"},
        {"location_before": "Dining Room", "action": "pull lever",
         "location_after": "Secret Passage"},
        {"location_before": "Secret Passage", "action": "crawl through tunnel",
         "location_after": "Kitchen"}
    ]

    # Answer 3: Action slightly misspelled, but path correct
    answer3_path = [
        {"location_before": "Entrance Hall", "action": "go narth",
         "location_after": "Dining Room"},  # Misspelled
        {"location_before": "Dining Room", "action": "go east",
         "location_after": "Kitchen"}
    ]

    # Answer 4: Incorrect action leading to wrong place or nowhere
    answer4_path = [
        {"location_before": "Entrance Hall", "action": "go west",
         "location_after": "Unknown Place"}
        # 'go west' is not valid from Entrance
    ]

    # Answer 5: Path structure incorrect (e.g. wrong start node)
    answer5_path = [
        {"location_before": "Dining Room", "action": "go east",
         "location_after": "Kitchen"}  # Starts at Dining Room
    ]

    # Answer 6: Correct path but with one invalid step in between
    answer6_path = [
        {"location_before": "Entrance Hall", "action": "go north",
         "location_after": "Dining Room"},
        {"location_before": "Dining Room", "action": "fly to moon",
         "location_after": "Moon Base"},  # Invalid action
        {"location_before": "Moon Base", "action": "go east",
         "location_after": "Kitchen"}
    ]

    # Answer 7: Empty path, but src == dst
    problem_src_eq_dst = {"src_node": "Kitchen", "dst_node": "Kitchen"}
    answer7_empty_path = []

    # Answer 8: Empty path, but src != dst
    problem_src_neq_dst_empty = {"src_node": "Entrance Hall",
                                 "dst_node": "Kitchen"}
    answer8_empty_path_wrong = []

    test_cases = [
        ("Correct Optimal", problem1, answer1_path),
        ("Correct Suboptimal", problem1, answer2_path),
        ("Action Misspelled", problem1, answer3_path),
        ("Incorrect Action", problem1, answer4_path),
        ("Incorrect Path Structure (Start Node)", problem1, answer5_path),
        ("Path with Invalid Middle Step", problem1, answer6_path),
        ("Empty Path, Src==Dst", problem_src_eq_dst, answer7_empty_path),
        ("Empty Path, Src!=Dst", problem_src_neq_dst_empty,
         answer8_empty_path_wrong),
    ]

    for name, problem, answer_path in test_cases:
        print(f"\n--- Validating: {name} ---")
        results = validate_rf_answer(problem, answer_path, game_map_graph)
        print(
            f"  Problem: From '{results['src_node']}' to '{results['dst_node']}'")
        print(f"  LLM Path Steps: {results['llm_path_steps']}")
        print(f"  Strict Action Score: {results['strict_action_score']:.2f}")
        print(f"  Loose Action Score: {results['loose_action_score']:.2f}")
        print(
            f"  Full Path Correct (Reasoning): {results['full_path_is_correct']}")

    # Example with custom path keys
    print("\n--- Validating with Custom Keys ---")
    custom_key_problem = {"src_node": "Entrance Hall",
                          "dst_node": "Dining Room"}
    custom_key_answer = [{"prev_loc": "Entrance Hall", "command": "go north",
                          "next_loc": "Dining Room"}]
    custom_keys_map = {"loc_before": "prev_loc", "action": "command",
                       "loc_after": "next_loc"}

    results_custom = validate_rf_answer(custom_key_problem, custom_key_answer,
                                        game_map_graph,
                                        path_keys=custom_keys_map)
    print(
        f"  Problem: From '{results_custom['src_node']}' to '{results_custom['dst_node']}'")
    print(f"  LLM Path Steps: {results_custom['llm_path_steps']}")
    print(
        f"  Strict Action Score: {results_custom['strict_action_score']:.2f}")
    print(f"  Loose Action Score: {results_custom['loose_action_score']:.2f}")
    print(
        f"  Full Path Correct (Reasoning): {results_custom['full_path_is_correct']}")
