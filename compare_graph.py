import json
from networkx.readwrite import json_graph
import networkx as nx

def load_llm_graph(json_file_path: str) -> nx.DiGraph:
    """
    从保存的 JSON 文件加载 LLM 构建的图
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def load_ground_truth_edges(zork_edges_path: str) -> nx.DiGraph:
    """
    从 zork1.edges.json 构建 Ground Truth 图
    """
    with open(zork_edges_path, "r") as f:
        edge_data = json.load(f)

    G = nx.DiGraph()
    for edge in edge_data:
        src = edge["src_node"].strip().lower()
        dst = edge["dst_node"].strip().lower()
        action = edge["action"].strip().lower()
        G.add_edge(src, dst, action=action)
    return G
def analyze_false_positives_with_gt_context_dict(false_positives, gt_graph):
    """
    对每个 false_positive 边，检查是否其 (src, dst) 节点对在 GT 中存在其他动作的边。
    返回结果为列表：每项是 (src, dst, predicted_action, true_actions OR 'NO EDGE')
    """
    """
    处理格式为 {"from": ..., "to": ..., "action": ...} 的 false positive 列表，
    并检查 GT 中是否存在相同 src-dst 的其他动作，或完全没有边。

    返回：
        List[Tuple(from, to, predicted_action, GT info)]
    """
    analysis = []

    for item in false_positives:
        src = item["from"].strip().lower()
        dst = item["to"].strip().lower()
        pred_action = item["action"].strip().lower()

        result = {
            "from": src,
            "to": dst,
            "action": pred_action
        }

        if gt_graph.has_edge(src, dst):
            true_action = gt_graph[src][dst]['action'].strip().lower()
            if true_action != pred_action:
                result["ground_truth_info"] = f"GT action = {true_action}"
            else:
                result["ground_truth_info"] = "GT action matches?!"
        else:
            alt_edges = [gt_graph[src][nbr]['action'] for nbr in gt_graph.successors(src)] if src in gt_graph else []
            reverse_edges = [gt_graph[pred][dst]['action'] for pred in gt_graph.predecessors(dst)] if dst in gt_graph else []
            msg = "NO EDGE" if not alt_edges and not reverse_edges else f"Other actions from src: {alt_edges}, to dst: {reverse_edges}"
            result["ground_truth_info"] = msg

        analysis.append(result)

    return str(analysis).replace("'",'"')

def compare_graphs(llm_G: nx.DiGraph, gt_G: nx.DiGraph):
    """
    比较两个图的边差异（基于 src, dst, action 三元组）
    """
    def edge_key(edge):
        return (edge[0].strip().lower(), edge[1].strip().lower(), edge[2]['action'].strip().lower())

    llm_edges = set(edge_key(e) for e in llm_G.edges(data=True))
    gt_edges = set(edge_key(e) for e in gt_G.edges(data=True))

    correct = llm_edges & gt_edges
    false_positives = llm_edges - gt_edges
    false_negatives = gt_edges - llm_edges
    print(len(false_positives),"false_positives")
    print(len(false_negatives),"false_negatives")
    print(len(correct),"correct")
    return {
        "correct": correct,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
from pathlib import Path
import json

json_files = Path('.').glob("llm_maze_graph_*.json")

fp_ratios = []
fn_ratios = []
precisions = []
recalls = []
f1s = []
accuracies = []
valid_list=['905', 'afflicted', 'anchor', 'ballyhoo', 'curses', 'cutthroat', 'dragon', 'enchanter',
 'enter', 'gold', 'hhgg', 'hollywood', 'huntdark', 'infidel', 'jewel', 'library', 'loose',
 'lostpig', 'ludicorp', 'lurking', 'partyfoul', 'pentari', 'planetfall', 'plundered',
 'sherlock', 'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'trinity',
 'yomomma', 'zenon', 'ztuu']
for json_file in json_files:
    graph = json_file.stem.replace("llm_maze_graph_", "")
    if graph in valid_list:
        print(f"正在处理图：{graph}")

        llm_G = load_llm_graph(str(json_file))
        gt_path = Path(f"D:/mango/data/{graph}/{graph}.edges.json")
        if not gt_path.exists():
            print(f"警告：找不到 ground truth 文件 {gt_path}，跳过")
            continue
        gt_G = load_ground_truth_edges(gt_path)

        diff = compare_graphs(llm_G, gt_G)
        tp = len(diff['correct'])               # True Positives
        fp = len(diff['false_positives'])       # False Positives
        fn = len(diff['false_negatives'])       # False Negatives

        # 获取节点数用于计算 TN
        nodes = set(llm_G.nodes) | set(gt_G.nodes)
        n = len(nodes)
        total_possible_edges = n * (n - 1)
        tn = total_possible_edges - (tp + fp + fn)

        total = tp + fp + fn + tn
        if total == 0:
            continue

        # 计算指标
        fp_ratios.append(fp / total)
        fn_ratios.append(fn / total)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)

        print(f"{graph}:")
        print(f"  False Positive 比例: {fp / total:.2%}")
        print(f"  False Negative 比例: {fn / total:.2%}")
        print(f"  精确率 (Precision): {precision:.2%}")
        print(f"  召回率 (Recall): {recall:.2%}")
        print(f"  F1 分数: {f1:.2%}")
        print(f"  准确率 (Accuracy): {accuracy:.2%}")
        print("")

# 平均值输出
if fp_ratios:
    print("=== 平均统计 ===")
    print(f"平均 False Positive 比例: {sum(fp_ratios)/len(fp_ratios):.2%}")
    print(f"平均 False Negative 比例: {sum(fn_ratios)/len(fn_ratios):.2%}")
    print(f"平均 精确率 (Precision): {sum(precisions)/len(precisions):.2%}")
    print(f"平均 召回率 (Recall): {sum(recalls)/len(recalls):.2%}")
    print(f"平均 F1 分数: {sum(f1s)/len(f1s):.2%}")
    print(f"平均 准确率 (Accuracy): {sum(accuracies)/len(accuracies):.2%}")
else:
    print("没有成功处理的图文件，无法计算平均值。")
