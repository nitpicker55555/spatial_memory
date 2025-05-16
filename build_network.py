import networkx as nx
from pathlib import Path

import re

from tqdm import tqdm

from chat_py import chat_single, message_template
from eval import read_walkthrough
import json,os
def add_edge_to_llm_graph(src_node: str, dst_node: str, action: str):
    """
    将 LLM 的输出边添加到构建中的图中。

    参数：
    - src_node: 起点位置名称（字符串）
    - dst_node: 终点位置名称（字符串）
    - action: 从 src 到 dst 的动作（例如 'north', 'up', 'enter'）
    """
    llm_graph.add_edge(src_node, dst_node, action=action)
    print_colored("edge added",'green')
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

def get_available_filename(filepath):
    """
    如果文件存在，则在文件名后添加数字，直到找到一个不存在的文件名。

    参数:
        filepath (str): 原始文件路径（包括文件名）

    返回:
        str: 可用的文件路径
    """
    if not os.path.exists(filepath):
        return filepath

    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        new_filename = f"{name}{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def extract_locations(text):
    previous_pattern = r'previous location:\s*(.*)'
    current_pattern = r'current location:\s*(.*)'

    previous_match = re.search(previous_pattern, text)
    current_match = re.search(current_pattern, text)

    previous_location = previous_match.group(1).strip() if previous_match else None
    current_location = current_match.group(1).strip() if current_match else None

    return previous_location, current_location


def validate_locations(json_locations, json_file_path):
    with open(json_file_path, 'r') as file:
        valid_locations = json.load(file)

    previous_location, current_location = json_locations['from'],json_locations['to']

    invalids = []

    if previous_location and previous_location not in valid_locations:
        invalids.append(f"'{previous_location}' is not a valid previous location.")
    if current_location and current_location not in valid_locations:
        invalids.append(f"'{current_location}' is not a valid current location.")

    if invalids:
        reference = f"Valid locations are: {', '.join(valid_locations)}."
        return " ".join(invalids) + " " + reference
    else:
        return None


def save_list_to_json(data_list, filename):
    """
    将列表保存为 JSON 文件
    :param data_list: 要保存的列表
    :param filename: 文件名（包含路径）
    """
    history_file_name=get_available_filename(filename)

    with open(history_file_name, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def load_list_from_json(filename):
    """
    从 JSON 文件读取数据并返回列表
    :param filename: 文件名（包含路径）
    :return: 读取的列表
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def eval_model(path_name:str,walkthrough:list):
    folder_name = os.path.basename(path_name)
    valid_map_list=[]
    location_path = os.path.join(path_name, f"{folder_name}.locations.json")
    sys_prompt="""
You are exploring a world described in natural language, step by step.

Your task is give back relative location change after each observation, showing only the previous location and the current location, using:
```json
 {"from": "west of House", "to": "south of House", "action": "south"}
 ```
 Please use lower case in the json response.
 Please notice, some actions may be not regular, but if location has been changed, you should take that action into json.
 
 if there is no location change, please reply 
 ```json
 {"action": "stay","from":None, "to":None}
 ```
 If the game is over, please reply
  ```json
 {"action": "quit","from":None, "to":None}
 ```
    """
    history_file_name_template="history_"+str(folder_name)+".json"
    # valid_name_template="valid_response_"+str(path_name.split('/')[-1])+".json"

    messages = [
        message_template('system',sys_prompt),

    ]

    for each_walk in tqdm(walkthrough):
        print_colored(str(each_walk),'blue')
        messages.append(message_template('user', each_walk))
        # valid_map_list.append(message_template('user', each_walk))

        response = chat_single(messages,verbose=True,mode='json')

        messages.append(message_template('assistant', str(response)))
        if 'to' in response:
            if response['to'] and response['from']:
                while True:
                    validation_result = validate_locations(response, location_path)
                    if not validation_result:
                        break  # 验证通过，跳出循环
                    print_colored(str(validation_result), 'red')

                    messages.append(message_template('user', validation_result))
                    response = chat_single(messages,verbose=True,mode='json')
                    messages.append(message_template('assistant', str(response)))
                try:
                    add_edge_to_llm_graph(response['from'],response['to'],response['action'])
                except Exception as e:
                    print_colored(e,'red')
    save_list_to_json(messages,history_file_name_template)


    # messages.append(message_template('user', str(walkthrough)))
    # response = chat_single(messages)
    # print(response)

BASE_DIR = Path("D:/mango/data")

for i, path_name in enumerate(BASE_DIR.iterdir()):
        if path_name.is_dir():
            if i>=1:
                # 初始化一个空的有向图（迷宫图是有方向的）
                llm_graph = nx.DiGraph()

                graph_arg = path_name.name
                DATA_DIR = path_name
                print(str(path_name))
                # 执行评估模型
                walkthrough_path = DATA_DIR / f"{graph_arg}.walkthrough"
                eval_model(str(path_name), walkthrough=read_walkthrough(walkthrough_path, 70))

                # 获取边信息
                edges_with_actions = list(llm_graph.edges(data=True))

                # 保存为 GraphML 格式
                graphml_path = f"llm_maze_graph_{graph_arg}.graphml"
                nx.write_graphml(llm_graph, graphml_path)

                # 保存为 JSON 格式
                json_path = f"llm_maze_graph_{graph_arg}.json"
                graph_json = nx.readwrite.json_graph.node_link_data(llm_graph)
                with open(json_path, "w") as f:
                    json.dump(graph_json, f, indent=2)