import json
import networkx as nx
from networkx.readwrite import json_graph

def convert_llm_graph_to_edges_format(json_file_path: str, output_path: str):
    # 加载图
    with open(json_file_path, "r") as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)

    # 转换为 zork1.edges.json 格式的列表
    edges_list = []
    for src, dst, attr in G.edges(data=True):
        edge = {
            "src_node": src,
            "dst_node": dst,
            "action": attr.get("action", ""),
            "seen_in_forward": -1,
            "seen_in_reversed": -1,
            "edge_min_step": -1,
            "seen_in_forward_answerable": -1,
            "seen_in_reversed_answerable": -1,
            "edge_min_step_answerable": -1
        }
        edges_list.append(edge)

    # 保存为 JSON 文件
    with open(output_path, "w") as f:
        json.dump(edges_list, f, indent=2)

    return output_path


from pathlib import Path

input_dir = Path("D:/spatial_memory")
output_dir = input_dir / "edges"
output_dir.mkdir(exist_ok=True)

# 遍历所有以 llm_maze_graph_ 开头的 JSON 文件
for input_file in input_dir.glob("llm_maze_graph_*.json"):
    graph_name = input_file.stem.replace("llm_maze_graph_", "")

    output_file = output_dir / f"{graph_name}.edges.json"

    convert_llm_graph_to_edges_format(str(input_file), str(output_file))
    print(f"已处理：{input_file.name} → {output_file.name}")

#
# # 执行转换
# output_edges_path = r"D:\spatial_memory\edges\edges_jewel.json"
# convert_llm_graph_to_edges_format(r"D:\spatial_memory\llm_maze_graph_jewel.json", output_edges_path)
