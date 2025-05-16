import json
import re
import numpy as np
import re
from collections import defaultdict
import pandas as pd
def generate_step_emojis(n):
    return [f"{i}\uFE0F\u20E3" for i in range(1, n + 1)]
# 假设文件名为 data.json，路径为 /mnt/data/data.json
json_file_path = "valid_response_night.json"

# 读取 JSON 文件并筛选出 role 为 assistant 的 content
with open(json_file_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# 生成 example_list
example_list = [item["content"] for item in data_list if item.get("role") == "assistant"]
#
# # Sample input (normally this would be a list of such strings)
# example_list = [
#     """movement label: horizontal
#
# ⬜ ⬜ ⬜
# ⬜ 🚩 🏃
# ⬜ ⬜ ⬜
#
# reasoning: i moved east within the maze of twisty passages from stop 1 to stop 2.
#
# previous location: maze of twisty passages (stop 1)
# current location: maze of twisty passages (stop 2)"""
# ]

# Emoji index list for labeling steps

def generate_step_labels(n):
    width = len(str(n))
    return [str(i + 1).zfill(width) for i in range(n)]

def parse_entry(entry):
    if "movement label: stay" in entry:
        return None

    movement_type = re.search(r"movement label: (\w+)", entry).group(1)
    prev_location = re.search(r"previous location: (.+)", entry).group(1).strip()
    curr_location = re.search(r"current location: (.+)", entry).group(1).strip()
    ascii_match = re.search(r"(⬜[⬜ 🚩🏃⬜]*\n[⬜ 🚩🏃⬜]*\n[⬜ 🚩🏃⬜]*)", entry)
    if not ascii_match:
        return None
    ascii_map = ascii_match.group(1).strip().split("\n")

    prev_pos = curr_pos = None
    for y, row in enumerate(ascii_map):
        cells = row.strip().split()
        for x, cell in enumerate(cells):
            if cell == '🚩':
                prev_pos = (x, y)
            elif cell == '🏃':
                curr_pos = (x, y)

    if prev_pos is None or curr_pos is None:
        return None

    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    return {
        'movement': movement_type,
        'prev_location': prev_location,
        'curr_location': curr_location,
        'delta': (dx, dy)
    }

parsed_data = [parse_entry(entry) for entry in example_list if parse_entry(entry)]
location_coords = {}
z_map = {}
step_sequence = []
step_labels = generate_step_labels(len(parsed_data) + 1)
vertical_links = {}

z_level = 0
first = parsed_data[0]['prev_location']
location_coords[first] = (0, 0)
z_map[first] = z_level
step_sequence.append((first, 0, 0, z_level))

for i, data in enumerate(parsed_data):
    prev = data['prev_location']
    curr = data['curr_location']
    dx, dy = data['delta']
    movement = data['movement']
    if movement == "vertical":
        z_level += 1
        vertical_links[z_level] = (step_labels[i], step_labels[i + 1])
    base_x, base_y = location_coords[prev]
    new_pos = (base_x + dx, base_y + dy)
    location_coords[curr] = new_pos
    z_map[curr] = z_level
    step_sequence.append((curr, new_pos[0], new_pos[1], z_level))

# 构建图并检测重叠
layers = defaultdict(dict)
overlaps = defaultdict(list)

for idx, (loc, x, y, z) in enumerate(step_sequence):
    key = (x, y)
    label = step_labels[idx]
    if key in layers[z]:
        overlaps[(x, y, z)].append(layers[z][key])  # 已有的
        overlaps[(x, y, z)].append(label)           # 新的
        # 合并显示
        layers[z][key] += f"/{label}"
    else:
        layers[z][key] = label

# 打印图并报告重叠
for z in sorted(layers.keys()):
    layer = layers[z]
    xs = [pos[0] for pos in layer]
    ys = [pos[1] for pos in layer]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    grid = [["⬜" for _ in range(width)] for _ in range(height)]

    for (x, y), label in layer.items():
        grid[y - min_y][x - min_x] = label

    header = f"Z-Level {z}:\n"
    if z in vertical_links:
        from_label, to_label = vertical_links[z]
        header += f"{to_label}\n---\n{from_label}\n"

    grid_str = "\n".join("  ".join(cell for cell in row) for row in grid)
    print(header + grid_str + "\n")

# 打印重叠报告
if overlaps:
    print("⚠️ 重叠警告：以下坐标在同一层出现多个步骤标签")
    for (x, y, z), labels in overlaps.items():
        unique_labels = sorted(set(labels))
        print(f"  Z-Level {z} @ ({x}, {y}): {' / '.join(unique_labels)}")
else:
    print("✅ 没有发现步骤重叠")

location_positions = defaultdict(set)
for (loc, x, y, z) in step_sequence:
    location_positions[loc].add((x, y, z))

# 查找重复位置的地名
conflicted_locations = {loc: positions for loc, positions in location_positions.items() if len(positions) > 1}

# 打印结果
if conflicted_locations:
    print("⚠️ 以下地名在不同坐标出现（可能存在逻辑错误）：")
    for loc, positions in conflicted_locations.items():
        formatted = ", ".join(f"(x={x}, y={y}, z={z})" for (x, y, z) in sorted(positions))
        print(f"  - {loc}: {formatted}")
else:
    print("✅ 所有地名都只出现在一个唯一坐标上")