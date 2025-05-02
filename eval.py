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
def model(prompt: str,sys_prompt,key_name='location') -> str:
    sys_prompt_ori="""
    please answer question in json format:
    {
    "%s":%s
    }
    
    """%(key_name,key_name)
    # print("sys_prompt",sys_prompt)
    print("prompt",prompt)
    # 创建消息
    messages = [
        message_template('system', sys_prompt_ori+sys_prompt),
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

def read_walkthrough(path: Path, max_steps: int = 70) -> str:
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
    return "\n".join(steps[:max_steps])
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
# --------- MAIN EVALUATION FUNCTION ---------
def evaluate_model_on_df_rf(TASK_TYPE):
    walkthrough = read_walkthrough(DATA_DIR / "night.walkthrough", MAX_STEP)
    actions = load_json(DATA_DIR / "night.actions.json")
    locations = load_json(DATA_DIR / "night.locations.json")
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

            src = task["src_node"]
            dst = task["dst_node"]

            actions_list = [step["action"] for step in task["path_details"]]
            sys_prompt=       (
                f"Walkthrough:\n{walkthrough}\n\n"
                f"The allowed actions are: {', '.join(actions)}.\n"
                                        )
            prompt = (

                f"The list of locations are: {', '.join(locations)}.\n\n"
                f"Starting from {src}, perform actions {actions_list}, where are you now?"
            )

            response = model(prompt,sys_prompt)['location']
            predicted = extract_last_location(response)
            print("predicted",predicted)
            print("dst",dst)
            if predicted == dst.lower():
                correct += 1
                judge_correct=True
            else:
                judge_correct=False
            total += 1
            each_json={
                "question":prompt,
                "response":response,
                "predicted":predicted,
                "dst":dst,
                "judge_correct":judge_correct,
                "easy":is_easy_df(task)

            }
            with open("results_each_desti_finding.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(each_json, ensure_ascii=False) + "\n")

    elif TASK_TYPE == "route_finding":
        tasks = load_jsonl(DATA_DIR / "night.all_pairs.jsonl")
        for task in tqdm(tasks):
            if task["min_step_total_answerable"] > MAX_STEP:
                continue

            src = task["src_node"]
            dst = task["dst_node"]
            sys_prompt=       (
                f"Walkthrough:\n{walkthrough}\n\n"
                f"The allowed actions are: {', '.join(actions)}.\n"
                f"The list of locations are: {', '.join(locations)}.\n\n"
                                        )
            prompt = (
                f"How can you go from {src} to {dst}?\n"
                f"Describe the trajectory in a Python list of dictionaries "
                f"with keys 'prev_node', 'action', and 'node'."
            )

            response = model(prompt,sys_prompt,key_name="trajectory")['trajectory']
            pred_path=None
            path_eval=is_path_valid(response, src, dst,maze_graph)
            if path_eval[0]:
                correct += 1
                judge_correct=True
            else:
                judge_correct=False

            total += 1
            print("predicted",response)
            print("dst",dst)
            each_json={
                "question":prompt,
                "response":response,
                "predicted":pred_path,
                "dst":dst,
                "judge_correct":judge_correct,
                "easy":is_easy_rf(task),
                "judge_reason": str(path_eval),

            }
            with open("results_each_route_finding.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(each_json, ensure_ascii=False) + "\n")

    return {"total": total, "correct": correct, "accuracy": correct / total if total else 0}

# Run evaluation
# # route_finding/desti_finding
data=evaluate_model_on_df_rf(TASK_TYPE="route_finding")
with open("results.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False) + "\n")

# walkthrough = read_walkthrough(DATA_DIR / "night.walkthrough", 70)
# print(walkthrough)