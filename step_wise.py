import json
import re
from collections import defaultdict
json_file_path = "valid_response_night2.json"
def print_colored(text, color='blue'):
    # ANSI颜色代码映射
    color_codes = {
        'black': '\033[30m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }

    # 获取颜色代码，如果没有就用默认蓝色
    color_code = color_codes.get(color.lower(), color_codes['blue'])
    reset_code = color_codes['reset']

    print(f"{color_code}{text}{reset_code}")

# 读取 JSON 文件并筛选出 role 为 assistant 的 content
with open(json_file_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# 生成 example_list
example_list = [item["content"] for item in data_list if item.get("role") == "assistant"]
story_line = [item["content"] for item in data_list if item.get("role") == "user"]

def summarize_text(text):
    lines = text.strip().splitlines()
    summary = lines[0] if lines else "<空>"
    return summary if len(summary) < 60 else summary[:57] + "..."
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

    dz = 0
    if movement_type == "vertical":
        if dy < 0:
            dz = 1  # upward
        elif dy > 0:
            dz = -1  # downward

    return {
        'movement': movement_type,
        'prev_location': prev_location,
        'curr_location': curr_location,
        'delta': (dx, dy),
        'dz': dz
    }
# 播放版本，直接输出每一步对应的原始 story_line 内容，并考虑 skipped entries（stay）

parsed_entries = []
entry_map = []  # maps parsed_data index to story_line index

for i, entry in enumerate(example_list):
    parsed = parse_entry(entry)
    if parsed:
        parsed_entries.append(parsed)
        entry_map.append(i)

step_labels = generate_step_labels(len(parsed_entries) + 1)

# 初始化
z_level = 0
location_coords = {}
z_map = {}
step_sequence = []
occupied_coords = defaultdict(dict)
location_positions = defaultdict(set)

# 初始点
first = parsed_entries[0]['prev_location']
location_coords[first] = (0, 0)
z_map[first] = z_level
step_sequence.append((step_labels[0], first, 0, 0, z_level))

# 播放构建路径
for i, data in enumerate(parsed_entries):
    story_idx = entry_map[i]
    content = story_line[story_idx]
    content_example_list = example_list[story_idx]

    label = step_labels[i + 1]
    movement = data['movement']
    prev = data['prev_location']
    curr = data['curr_location']
    dx, dy = data['delta']

    dz = data['dz']
    if movement == "vertical":
        z_level += dz
    base_x, base_y = location_coords[prev]
    new_pos = (base_x + dx, base_y + dy)

    location_coords[curr] = new_pos
    z_map[curr] = z_level
    step_sequence.append((label, curr, new_pos[0], new_pos[1], z_level))

    print(f"\n====================")
    print(f"Step {label}: {curr}")
    print(f"Position: (x={new_pos[0]}, y={new_pos[1]}, z={z_level})")
    print(f"Action: {movement.upper()}")
    print_colored(f"Story line content:\n{content}",'blue')
    print_colored(f"content_example_list line content:\n{content_example_list}",'blue')

    if new_pos in occupied_coords[z_level]:
        conflict_label = occupied_coords[z_level][new_pos]
        # print(f"⚠️ 坐标重叠：位置 {new_pos} 已被 Step {conflict_label} 占用！")

    if (new_pos[0], new_pos[1], z_level) not in location_positions[curr]:
        if curr in location_positions and len(location_positions[curr]) > 0:
            print_colored(f"⚠️ 地名重复：'{curr}' 已在其他坐标出现过！",'red')
        location_positions[curr].add((new_pos[0], new_pos[1], z_level))

    occupied_coords[z_level][new_pos] = label
    layer = occupied_coords[z_level]
    xs = [x for (x, y) in layer]
    ys = [y for (x, y) in layer]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    grid = [["⬜" for _ in range(width)] for _ in range(height)]
    for (x, y), lbl in layer.items():
        grid[y - min_y][x - min_x] = lbl

    print(f"\n🗺️ 当前 Z-Level {z_level} 地图：")
    print("\n".join("  ".join(cell for cell in row) for row in grid))
