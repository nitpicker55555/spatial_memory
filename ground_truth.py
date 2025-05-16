import json
import re
from collections import defaultdict
json_file_path = "valid_response_night2.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)
example_list = [item["content"] for item in data_list if item.get("role") == "assistant"]
story_line = [item["content"] for item in data_list if item.get("role") == "user"]

# 有效的方向
valid_actions = [
    "down", "south", "north", "southwest", "northeast",
    "west", "go panel", "east", "follow mouse", "up"
]

# 地理方向映射，忽略无空间含义的指令
direction_map = {
    "north": (-1, 0),
    "up": (-1, 0),  # 视为 north
    "south": (1, 0),
    "down": (1, 0),  # 视为 south
    "east": (0, 1),
    "west": (0, -1),
    "northeast": (-1, 1),
    "southwest": (1, -1)
}

# 抽取 ACT 列表
def extract_actions(logs):
    actions = []
    for log in logs:
        for line in log.splitlines():
            if line.startswith("==>ACT:"):
                action = line.split("==>ACT:")[1].strip()
                actions.append(action)
    return actions

# 分析 emoji 的移动方向
def parse_position(grid):
    lines = grid.strip().splitlines()
    pos_flag = None
    pos_runner = None
    for r, line in enumerate(lines):
        for c, ch in enumerate(line.split()):
            if ch == "🚩":
                pos_flag = (r, c)
            elif ch == "🏃":
                pos_runner = (r, c)

    if pos_flag is None:
        print(grid)
        print("⚠️ 未找到 🚩（起点）的位置。")
    if pos_runner is None:
        print(grid)
        print("⚠️ 未找到 🏃（目标）的位置。")

    return pos_flag, pos_runner


# 主处理函数
def analyze_movements(action_logs, movement_logs):
    actions = extract_actions(action_logs)

    for idx, (action, log) in enumerate(zip(actions, movement_logs)):
        # 只处理空间相关方向
        if action not in direction_map:
            continue

        dir_row, dir_col = direction_map[action]

        # 获取 emoji 网格部分
        lines = log.splitlines()
        grid_lines = lines[1:4]  # 假设固定为第二到第四行是地图
        grid_str = "\n".join(lines)
        flag_pos, runner_pos = parse_position(grid_str)

        if not flag_pos or not runner_pos:
            print(action_logs[idx],movement_logs[idx])

            print(f"[{idx}] 缺失🚩或🏃位置，跳过")
            continue

        expected_runner_pos = (flag_pos[0] + dir_row, flag_pos[1] + dir_col)

        if runner_pos != expected_runner_pos:
            print(action_logs[idx],movement_logs[idx])
            print(f"❌ 第 {idx} 项动作 `{action}` 错误")
            print(f"🏃 实际位置: {runner_pos}, 预期位置: {expected_runner_pos}")
            print(f"🚩位置: {flag_pos}")
            print("-----")

# 示例数据
# story_line = [
#     "==>STEP NUM: 3\n==>ACT: east\n==>OBSERVATION: Hall Outside Elevator",
#     "==>STEP NUM: 4\n==>ACT: north\n==>OBSERVATION: Some Room"
# ]

movement_logs = [
    "movement label: horizontal\n\n⬜ ⬜ ⬜  \n⬜ 🚩 🏃  \n⬜ ⬜ ⬜  \n\nreasoning: i moved east...",
    "movement label: vertical\n\n⬜ 🏃 ⬜  \n⬜ 🚩 ⬜  \n⬜ ⬜ ⬜  \n\nreasoning: i moved north..."
]

analyze_movements(story_line, example_list)
