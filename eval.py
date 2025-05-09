import json
from pathlib import Path
from typing import List, Dict
import re

from tqdm import tqdm

from chat_py import chat_single, message_template
import networkx as nx
# --------- CONFIG ---------
DATA_DIR = Path(r"D:/mango/data/night")
# TASK_TYPE = "desti_finding"  # or "route_finding"
MAX_STEP = 70

# Dummy model function interface
def model(walkthrough:list,prompt: str,sys_prompt,key_name='location') -> str:
    sys_prompt_ori="""
    please answer question in json format:
    {
    "%s":%s
    }
    
    """%(key_name,key_name)
    print("sys_prompt",sys_prompt)
    print("prompt",prompt)
    # 创建消息
    messages = [
        message_template('system', sys_prompt_ori+sys_prompt + ". walkthrough: "+str(walkthrough)),
        message_template('user', prompt)
    ]

    # 发送请求
    response = chat_single(messages,mode='json')
    print(response)
    return response
# --------- HELPERS ---------
def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_walkthrough(path: Path, max_steps: int = 70) -> list:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    steps = []
    current = []
    started = False  # 标记是否开始遇到有效的 STEP

    for line in lines:
        if line.startswith("==>STEP NUM:"):
            if started and current:
                steps.append("".join(current))
                current = []
            started = True  # 开始记录 step 内容
        if started:
            current.append(line)

    if current:
        steps.append("".join(current))
    return steps[:max_steps]
    # return "\n".join(steps[:max_steps])

def extract_last_location(response: str) -> str:
    match = re.findall(r"(?:You are in|You are at|Location:)\s(.+)", response)
    return match[-1].strip().lower() if match else response.strip().lower()

def is_path_valid(pred_path: List[Dict], src: str, dst: str,maze_graph) -> bool:
    try:
        if not pred_path:
            return False, "Path is empty."

        if pred_path[0]["prev_node"].lower() != src.lower():
            return False, f"Path starts at {pred_path[0]['prev_node']}, expected {src}."

        if pred_path[-1]["node"].lower() != dst.lower():
            return False, f"Path ends at {pred_path[-1]['node']}, expected {dst}."

        for i, step in enumerate(pred_path):
            a = step["prev_node"].lower()
            b = step["node"].lower()
            action = step["action"].lower()

            if not maze_graph.has_edge(a, b):
                return False, f"Invalid edge from {a} to {b} (missing in graph)."
            if maze_graph[a][b]["action"] != action:
                return False, f"Action mismatch on edge {a} -> {b}: expected '{maze_graph[a][b]['action']}', got '{action}'."

        return True, None
    except Exception as e:
        return False, f"Exception during validation: {str(e)}"

def is_easy_df(data):
    path_details = data.get("path_details", [])
    is_easy = True

    # 判断路径中是否有任何一条边是 reverse-only 或未出现（即 seen_in_forward == 9999）
    for edge in path_details:
        forward_seen = edge.get("seen_in_forward_answerable", 9999)
        if forward_seen == 9999:
            is_easy = False
            break
    return is_easy
def is_easy_rf(min_step_forward_answerable):
    """
    判断一条 RF 样本是否为 hard。
    如果整个路径的 min_step_forward_answerable == 9999，则为 hard。
    """
    return min_step_forward_answerable != 9999
def single_desti_finding(task):
    walkthrough = read_walkthrough(DATA_DIR / "night.walkthrough", MAX_STEP)
    actions = load_json(DATA_DIR / "night.actions.json")
    locations = load_json(DATA_DIR / "night.locations.json")


    src = task["src_node"]
    dst = task["dst_node"]

    actions_list = [step["action"] for step in task["path_details"]]
    sys_prompt = (
        f"The allowed actions are: {', '.join(actions)}.\n"
    )
    prompt = (

        f"The list of locations are: {', '.join(locations)}.\n\n"
        f"Starting from {src}, perform actions {actions_list}, where are you now?"
    )

    response = model(walkthrough,prompt, sys_prompt)['location']
    predicted = extract_last_location(response)

    if predicted == dst.lower():

        judge_correct = True
    else:
        judge_correct = False
    print("predicted", predicted)
    print("dst", dst)
    print("judge_correct",judge_correct)

    each_json = {
        "question": prompt,
        "response": response,
        "predicted": predicted,
        "dst": dst,
        "judge_correct": judge_correct,
        "easy": is_easy_df(task)

    }
    with open("results_each_desti_finding.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(each_json, ensure_ascii=False) + "\n")




# If the graph is already built, draw it

def single_route_finding(task):
    walkthrough = read_walkthrough(DATA_DIR / "night.walkthrough", MAX_STEP)
    actions = load_json(DATA_DIR / "night.actions.json")
    locations = load_json(DATA_DIR / "night.locations.json")
    edges_data = load_json(DATA_DIR / "night.edges.json")
    maze_graph = nx.DiGraph()
    for edge in edges_data:
        src = edge["src_node"].lower()
        dst = edge["dst_node"].lower()
        action = edge["action"].lower()
        maze_graph.add_edge(src, dst, action=action)
    correct = 0
    total = 0
    src = task["src_node"]
    dst = task["dst_node"]
    sys_prompt = (
        f"The allowed actions are: {', '.join(actions)}.\n"
        f"The list of locations are: {', '.join(locations)}.\n\n"
    )
    prompt = (
        f"How can you go from {src} to {dst}?\n"
        f"Describe the trajectory in a Python list of dictionaries "
        f"with keys 'prev_node', 'action', and 'node'."
    )

    response = model(walkthrough, prompt, sys_prompt, key_name="trajectory")['trajectory']
    pred_path = None
    path_eval = is_path_valid(response, src, dst, maze_graph)
    if path_eval[0]:
        correct += 1
        judge_correct = True
    else:
        judge_correct = False

    total += 1
    print("predicted", response)
    print("dst", dst)
    print("judge_correct",judge_correct)
    each_json = {
        "question": prompt,
        "response": response,
        "predicted": pred_path,
        "dst": dst,
        "judge_correct": judge_correct,
        "easy": is_easy_rf(task),
        "judge_reason": str(path_eval),

    }
    with open("results_each_route_finding.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(each_json, ensure_ascii=False) + "\n")
# --------- MAIN EVALUATION FUNCTION ---------
def evaluate_model_on_df_rf(TASK_TYPE):

    edges_data = load_json(DATA_DIR / "night.edges.json")

    # Build maze graph (directed)
    maze_graph = nx.DiGraph()
    for edge in edges_data:
        src = edge["src_node"].lower()
        dst = edge["dst_node"].lower()
        action = edge["action"].lower()
        maze_graph.add_edge(src, dst, action=action)
    correct = 0
    total = 0
    judge_correct=False
    if TASK_TYPE == "desti_finding":
        tasks = load_jsonl(DATA_DIR / "night.all2all.jsonl")
        for task in tqdm(tasks):
            if task["min_step_total_answerable"] > MAX_STEP:
                continue
            single_desti_finding(task)

    elif TASK_TYPE == "route_finding":
        tasks = load_jsonl(DATA_DIR / "night.all_pairs.jsonl")
        for task in tqdm(tasks):
            if task["min_step_total_answerable"] > MAX_STEP:
                continue

            single_route_finding(task)

    # return {"total": total, "correct": correct, "accuracy": correct / total if total else 0}

# Run evaluation
# # route_finding/desti_finding
# evaluate_model_on_df_rf(TASK_TYPE="route_finding")
# single_desti_finding({"src_node": "computer site", "dst_node": "maze of twisty passages (stop 2)", "diff_shortest": 0, "path_details": [{"prev_node": "computer site", "node": "hall outside computer site", "action": "northeast", "seen_in_forward": 1, "seen_in_reversed": 53, "edge_min_step": 1, "seen_in_forward_answerable": 1, "seen_in_reversed_answerable": 53, "edge_min_step_answerable": 1}, {"prev_node": "hall outside computer site", "node": "stairwell (third floor)", "action": "west", "seen_in_forward": 9, "seen_in_reversed": 52, "edge_min_step": 9, "seen_in_forward_answerable": 9, "seen_in_reversed_answerable": 52, "edge_min_step_answerable": 9}, {"prev_node": "stairwell (third floor)", "node": "stairwell (second floor)", "action": "down", "seen_in_forward": 10, "seen_in_reversed": 51, "edge_min_step": 10, "seen_in_forward_answerable": 10, "seen_in_reversed_answerable": 51, "edge_min_step_answerable": 10}, {"prev_node": "stairwell (second floor)", "node": "outside physics office", "action": "east", "seen_in_forward": 23, "seen_in_reversed": 50, "edge_min_step": 23, "seen_in_forward_answerable": 23, "seen_in_reversed_answerable": 50, "edge_min_step_answerable": 23}, {"prev_node": "outside physics office", "node": "hall (2nd floor, middle of north/south hall)", "action": "south", "seen_in_forward": 24, "seen_in_reversed": 49, "edge_min_step": 24, "seen_in_forward_answerable": 24, "seen_in_reversed_answerable": 49, "edge_min_step_answerable": 24}, {"prev_node": "hall (2nd floor, middle of north/south hall)", "node": "maze of twisty passages (stop 1)", "action": "down", "seen_in_forward": 64, "seen_in_reversed": 48, "edge_min_step": 48, "seen_in_forward_answerable": 64, "seen_in_reversed_answerable": 48, "edge_min_step_answerable": 48}, {"prev_node": "maze of twisty passages (stop 1)", "node": "maze of twisty passages (stop 2)", "action": "east", "seen_in_forward": 30, "seen_in_reversed": 47, "edge_min_step": 30, "seen_in_forward_answerable": 30, "seen_in_reversed_answerable": 47, "edge_min_step_answerable": 30}], "step_count": 7, "id": "eab5cd82effc87bb07a67209b36cee7b06812850e96292026657db2f76860ae1", "min_step_forward": 64, "min_step_total": 48, "min_step_forward_answerable": 64, "min_step_total_answerable": 48}
# )

# walkthrough = read_walkthrough(DATA_DIR / "night.walkthrough", 70)
# for i in walkthrough:
#     print("????????????????????????")
#     print(i)