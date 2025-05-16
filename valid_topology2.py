import json
import os
from collections import defaultdict
from pathlib import Path
from collections import Counter
# ---------------------------
# Conflict 检测逻辑
# ---------------------------

# 1. Naming Conflict: 多个节点有相同的名字
def detect_conflicts_with_roundtrip(edges, locations):
    from collections import defaultdict

    # 基础图构建
    forward_edges = defaultdict(dict)  # (src)[action] = dst
    for edge in edges:
        forward_edges[edge["src_node"]][edge["action"]] = edge["dst_node"]

    # 命名冲突
    name_to_nodes = defaultdict(set)
    for loc in locations:
        name = loc.lower().strip()
        name_to_nodes[name].add(loc)
    naming_conflict_nodes = {loc for nodes in name_to_nodes.values() if len(nodes) > 1 for loc in nodes}

    # 方向冲突
    directional_conflict_keys = defaultdict(set)
    for edge in edges:
        key = (edge["src_node"], edge["action"])
        directional_conflict_keys[key].add(edge["dst_node"])
    directional_conflict_keys = {k for k, v in directional_conflict_keys.items() if len(v) > 1}

    # 单步拓扑冲突
    reverse_lookup = {
        "north": "south", "south": "north",
        "east": "west", "west": "east",
        "up": "down", "down": "up",
        "northeast": "southwest", "southwest": "northeast",
        "northwest": "southeast", "southeast": "northwest"
    }
    topology_conflicts = {}
    for src, actions in forward_edges.items():
        for action, dst in actions.items():
            rev = reverse_lookup.get(action)
            if not rev:
                continue

            # 先尝试获取反向方向的边
            reverse_dst = forward_edges.get(dst, {}).get(rev)

            if reverse_dst:
                # 存在反向边，但目标错误
                if reverse_dst != src:
                    topology_conflicts[(src, dst, action)] = {
                        "expected_reverse": f"{dst} -- {rev} --> {src}",
                        "actual_reverse": f"{dst} -- {rev} --> {reverse_dst}"
                    }
            else:
                # 如果反向边缺失，则尝试查找是否有任何边回到 src，但方向不对
                back_found = False
                for back_action, back_target in forward_edges.get(dst,
                                                                  {}).items():
                    if back_target == src:
                        back_found = True
                        # 如果方向也属于 reverse_lookup，但不匹配
                        if back_action in reverse_lookup and back_action != rev:
                            topology_conflicts[(src, dst, action)] = {
                                "expected_reverse": f"{dst} -- {rev} --> {src}",
                                "actual_reverse": f"{dst} -- {back_action} --> {src}"
                            }
                        break
                # elif not reverse_dst:
                #     topology_conflicts[(src, dst, action)] = {
                #         "expected_reverse": f"{dst} -- {rev} --> {src}",
                #         "actual_reverse": "None"
                #     }

    # 闭环检测（4步闭环 west → east → south → north）
    roundtrip_conflicts = []
    sequences = [("west", "east", "south", "north")]
    for start in forward_edges:
        for seq in sequences:
            a = forward_edges.get(start, {}).get(seq[0])
            b = forward_edges.get(a, {}).get(seq[1]) if a else None
            c = forward_edges.get(b, {}).get(seq[2]) if b else None
            end = forward_edges.get(c, {}).get(seq[3]) if c else None
            if end and end != start:
                roundtrip_conflicts.append({
                    "start": start,
                    "end": end,
                    "path": " → ".join(seq),
                    "trace": [start, a, b, c, end]
                })

    # 汇总冲突边并附加原因
    conflict_edges = []
    for edge in edges:
        src, dst, action = edge["src_node"], edge["dst_node"], edge["action"]
        name_conflict = src in naming_conflict_nodes or dst in naming_conflict_nodes
        directional_conflict = (src, action) in directional_conflict_keys
        topology_key = (src, dst, action)
        topology_conflict = topology_key in topology_conflicts

        if name_conflict or directional_conflict or topology_conflict:
            edge_copy = edge.copy()
            edge_copy['conflict_types'] = []
            edge_copy['conflict_reason'] = []

            if name_conflict:
                edge_copy['conflict_types'].append("naming")
                edge_copy['conflict_reason'].append("Duplicate or ambiguous location name")

            if directional_conflict:
                edge_copy['conflict_types'].append("directional")
                # 保留 key → set of dst 的映射，只过滤满足冲突条件的条目
                # 创建一个字典，记录每个 (src_node, action) 对应的所有目标节点
                directional_conflict_raw = defaultdict(set)
                for edge in edges:
                    key = (edge["src_node"], edge["action"])
                    directional_conflict_raw[key].add(edge["dst_node"])

                # 只保留那些指向多个目的地的冲突项
                directional_conflict_keys = {
                    k: v for k, v in directional_conflict_raw.items() if
                    len(v) > 1
                }

                destinations = directional_conflict_keys[(src, action)]
                detail = f"{src} -- {action} --> {', '.join(destinations)}"
                edge_copy['conflict_reason'].append(
                    f"Multiple destinations for same direction: {detail}")
                print( f"Multiple destinations for same direction: {detail}")
            if topology_conflict:
                edge_copy['conflict_types'].append("topology")
                reason = topology_conflicts[topology_key]
                detail = f"Expected: {reason['expected_reverse']}; Actual: {reason['actual_reverse']}"
                edge_copy['conflict_reason'].append(detail)

            conflict_edges.append(edge_copy)
    # print("roundtrip_conflicts",roundtrip_conflicts)
    return conflict_edges, roundtrip_conflicts

# 函数返回两个列表：
# - conflict_edges: 标准三类冲突
# - roundtrip_conflicts: 新增的4步闭环不一致冲突列表



def extract_conflict_pairs(conflict_edges):
    """
    提取冲突边中涉及的节点对（src, dst）以及冲突类型
    """
    conflict_pairs = set()
    for edge in conflict_edges:
        src = edge["src_node"]
        dst = edge["dst_node"]
        for conflict_type in edge.get("conflict_types", []):
            conflict_pairs.add((src, dst, conflict_type))
    return list(conflict_pairs)


# path_name="D:/mango/data/night"
# 模拟路径：你应将其指向解压后的 data 文件夹

base_path = r"D:\spatial_memory\edges"
results = {}
valid_list=['905', 'afflicted', 'anchor', 'ballyhoo', 'curses', 'cutthroat', 'dragon', 'enchanter',
 'enter', 'gold', 'hhgg', 'hollywood', 'huntdark', 'infidel', 'jewel', 'library', 'loose',
 'lostpig', 'ludicorp', 'lurking', 'partyfoul', 'pentari', 'planetfall', 'plundered',
 'sherlock', 'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'trinity',
 'yomomma', 'zenon', 'ztuu']
# 遍历 base_path 下的所有子文件夹和其中的 *.edges.json 文件
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".edges.json"):
            full_path = os.path.join(root, file)
            folder_key = file.replace(".edges.json", "")
            if folder_key in valid_list:
                print("Current file:", full_path)
                print("Folder key:", folder_key)

                data_dir = Path(full_path)


                edges_path = full_path
                locations_path = f"D:/mango/data/{folder_key}/{folder_key}.locations.json"

                with open(edges_path, "r") as f:
                    edges = json.load(f)

                with open(locations_path, "r") as f:
                    locations = json.load(f)

                conflict_edges,round_conflict = detect_conflicts_with_roundtrip(edges, locations)
                # print("conflict_edges",conflict_edges)
                print("round_conflict",round_conflict)
                # === 新增部分：统计信息 ===
                conflict_type_counter = Counter()
                for conflict in conflict_edges:
                    conflict_type_counter.update(conflict.get("conflict_types", []))

                print(f"→ Total conflicts in: {len(conflict_edges)}")
                print("→ Conflict type counts:")
                for conflict_type, count in conflict_type_counter.items():
                    print(f"   - {conflict_type}: {count}")

                # 保存到结果中，包括冲突列表和统计信息
                results[folder_key] = {
                    "conflict_count": len(conflict_edges),
                    "conflict_type_counts": dict(conflict_type_counter),
                    "conflicts": conflict_edges
                }

# 写入总结果
output_path = os.path.join(base_path, "all_conflict_edges.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")