import json


def _load_jsonl_data(file_path):
    """Helper function to load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
        raise
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")
        raise
    return data


def analyze_and_extract_problems(game_name,
                                 step_num_threshold):
    all_pairs_path = fr"D:\data\data\data\{game_name}\{game_name}.all_pairs.jsonl"  # 示例路径
    all2all_path = fr"D:\data\data\data\{game_name}\{game_name}.all2all.jsonl"
    """
    分析 RF (route-finding) 和 DF (destination-finding) 问题中的 "easy" 和 "hard" 数量，
    并提取这些问题本身。基于 utils/get_map_statistic.py 中的逻辑。

    Args:
        all_pairs_path (str): X.all_pairs.jsonl 文件的路径 (用于 RF 问题)。
        all2all_path (str): X.all2all.jsonl 文件的路径 (用于 DF 问题)。
        step_num_threshold (int): 用于区分 "easy" 和 "hard" 的步数阈值。
                                  对应 get_map_statistic.py 中的 'step_num' 参数。

    Returns:
        dict: 一个包含 RF 和 DF 问题中 "easy", "hard" 数量以及对应问题列表的字典。
              问题列表中的每个元素都是从原始 JSONL 文件解析得到的字典。
              如果文件读取或处理失败，则返回 None。
              示例:
              {
                  "route_finding": {
                      "total_entries": 0, "easy_count": 0, "hard_count": 0,
                      "not_classified_count": 0, "missing_fields_count": 0,
                      "easy_problems": [problem_dict_1, ...],
                      "hard_problems": [problem_dict_2, ...]
                  },
                  "desti_finding": {
                      "total_entries": 0, "easy_count": 0, "hard_count": 0,
                      "not_classified_count": 0, "missing_fields_count": 0,
                      "easy_problems": [problem_dict_3, ...],
                      "hard_problems": [problem_dict_4, ...]
                  }
              }
    """
    results = {
        "route_finding": {
            "total_entries": 0, "easy_count": 0, "hard_count": 0,
            "not_classified_count": 0, "missing_fields_count": 0,
            "easy_problems": [], "hard_problems": []
        },
        "desti_finding": {
            "total_entries": 0, "easy_count": 0, "hard_count": 0,
            "not_classified_count": 0, "missing_fields_count": 0,
            "easy_problems": [], "hard_problems": []
        }
    }

    # 处理 Route Finding (RF) 问题 from all_pairs.jsonl
    try:
        rf_problems_data = _load_jsonl_data(all_pairs_path)
        results["route_finding"]["total_entries"] = len(rf_problems_data)
        for problem in rf_problems_data:
            if "min_step_forward_answerable" not in problem or \
                    "min_step_total_answerable" not in problem:
                results["route_finding"]["missing_fields_count"] += 1
                continue

            msfa = problem["min_step_forward_answerable"]
            msta = problem["min_step_total_answerable"]

            if not isinstance(msfa, (int, float)) or not isinstance(msta, (
            int, float)):
                results["route_finding"]["missing_fields_count"] += 1
                continue

            if msfa <= step_num_threshold:
                results["route_finding"]["easy_count"] += 1
                results["route_finding"]["easy_problems"].append(problem)
            elif msta <= step_num_threshold:  # Implies msfa > step_num_threshold
                results["route_finding"]["hard_count"] += 1
                results["route_finding"]["hard_problems"].append(problem)
            else:
                results["route_finding"]["not_classified_count"] += 1

    except Exception as e:
        print(f"处理 RF 问题时发生错误: {e}")
        return None

    # 处理 Destination Finding (DF) 问题 from all2all.jsonl
    try:
        df_problems_data = _load_jsonl_data(all2all_path)
        results["desti_finding"]["total_entries"] = len(df_problems_data)
        for problem in df_problems_data:
            if "min_step_forward_answerable" not in problem or \
                    "min_step_total_answerable" not in problem:
                results["desti_finding"]["missing_fields_count"] += 1
                continue

            msfa = problem["min_step_forward_answerable"]
            msta = problem["min_step_total_answerable"]

            if not isinstance(msfa, (int, float)) or not isinstance(msta, (
            int, float)):
                results["desti_finding"]["missing_fields_count"] += 1
                continue

            if msta <= step_num_threshold:
                if msfa <= step_num_threshold:
                    results["desti_finding"]["easy_count"] += 1
                    results["desti_finding"]["easy_problems"].append(problem)
                else:  # msfa > step_num_threshold
                    results["desti_finding"]["hard_count"] += 1
                    results["desti_finding"]["hard_problems"].append(problem)
            else:
                results["desti_finding"]["not_classified_count"] += 1

    except Exception as e:
        print(f"处理 DF 问题时发生错误: {e}")
        return None

    return results
if __name__ == "__main__":

# 使用示例：
# 请将下面的路径替换为您的实际文件路径和合适的阈值

    threshold = 53 # 示例阈值
    game_name='curses'
    analysis_results = analyze_and_extract_problems(game_name, threshold)

    if analysis_results:
        print(f"--- 基于 utils/get_map_statistic.py 逻辑和阈值 {threshold} 步 ---")

        print(f"\n路径查找 (RF) 问题:")
        print(f"  总条目数: {analysis_results['route_finding']['total_entries']}")
        print(f"  Easy 数量: {analysis_results['route_finding']['easy_count']}")
        print(f"  Hard 数量: {analysis_results['route_finding']['hard_count']}")
        print(f"  未分类数量: {analysis_results['route_finding']['not_classified_count']}")
        print(f"  缺少字段数量: {analysis_results['route_finding']['missing_fields_count']}")

        # 如果需要将 RF easy 问题保存为 JSON 文件
        with open("rf_easy_problems.json", "w", encoding="utf-8") as f:
            json.dump(analysis_results['route_finding']['easy_problems'], f, indent=4, ensure_ascii=False)
        print(f"  RF Easy 问题列表已提取 (共 {len(analysis_results['route_finding']['easy_problems'])} 个)")

        # 如果需要将 RF hard 问题保存为 JSON 文件
        with open("rf_hard_problems.json", "w", encoding="utf-8") as f:
            json.dump(analysis_results['route_finding']['hard_problems'], f, indent=4, ensure_ascii=False)
        print(f"  RF Hard 问题列表已提取 (共 {len(analysis_results['route_finding']['hard_problems'])} 个)")

        print("-" * 20)
        print(f"\n目的地查找 (DF) 问题:")
        print(f"  总条目数: {analysis_results['desti_finding']['total_entries']}")
        print(f"  Easy 数量: {analysis_results['desti_finding']['easy_count']}")
        print(f"  Hard 数量: {analysis_results['desti_finding']['hard_count']}")
        print(f"  未分类数量: {analysis_results['desti_finding']['not_classified_count']}")
        print(f"  缺少字段数量: {analysis_results['desti_finding']['missing_fields_count']}")

        # 如果需要将 DF easy 问题保存为 JSON 文件
        with open("df_easy_problems.json", "w", encoding="utf-8") as f:
            json.dump(analysis_results['desti_finding']['easy_problems'], f, indent=4, ensure_ascii=False)
        print(f"  DF Easy 问题列表已提取 (共 {len(analysis_results['desti_finding']['easy_problems'])} 个)")

        # 如果需要将 DF hard 问题保存为 JSON 文件
        with open("df_hard_problems.json", "w", encoding="utf-8") as f:
            json.dump(analysis_results['desti_finding']['hard_problems'], f, indent=4, ensure_ascii=False)
        print(f"  DF Hard 问题列表已提取 (共 {len(analysis_results['desti_finding']['hard_problems'])} 个)")

    else:
        print("未能成功分析问题。")