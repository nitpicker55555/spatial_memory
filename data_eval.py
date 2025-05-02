import json

# 文件路径（请根据实际文件位置调整）
file_path = r'D:\mango\data\night\night.all_pairs.jsonl'  # 示例路径，替换为你的文件


hard_count_rf = 0
total_count_rf = 0

def is_hard_df(path_details):
    for edge in path_details:
        if edge.get("seen_in_forward_answerable", 9999) == 9999:
            return True
    return False
# 读取 JSONL 文件并判断每条记录是否为 hard
with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        min_step_forward_answerable = data.get("min_step_forward_answerable",
                                               9999)
        total_count_rf += 1
        if min_step_forward_answerable == 9999:
            hard_count_rf += 1

print( hard_count_rf, total_count_rf, hard_count_rf / total_count_rf if total_count_rf else 0)