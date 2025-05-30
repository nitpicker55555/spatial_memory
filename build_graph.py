import re

from tqdm import tqdm

from chat_py import chat_single, message_template
from eval import read_walkthrough
import json,os
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


def validate_locations(text, json_file_path):
    with open(json_file_path, 'r') as file:
        valid_locations = json.load(file)

    previous_location, current_location = extract_locations(text)

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

Your task is to construct a 3x3 ASCII-style emoji grid after each observation, showing only the previous location and the current location, using:

🚩 to represent your previous location (always in the center of the grid)

🏃 to represent your current location, positioned relative to 🚩 according to your most recent action

Based on the direction described, label the movement type as one of:

HORIZONTAL if the movement is across a 2D plane (e.g., north, east)

VERTICAL if the movement is up or down

STAY if there was no movement

The emoji grid must reflect the correct spatial relation:

For horizontal moves, place 🏃 in the corresponding direction:

north → top center

south → bottom center

east → center right

west → center left

northeast → top right

northwest → top left

southeast → bottom right

southwest → bottom left

For vertical moves, place 🏃 either:

top center (↑) if moved up

bottom center (↓) if moved down
(this vertical grid does not use compass directions but only a Z-axis relation)

For STAY, place 🏃 directly over 🚩 in the center

After each observation, print:

Movement label: HORIZONTAL, VERTICAL, or STAY

A 3x3 grid of emojis showing the movement

A short reasoning sentence (e.g., "I climbed up from the basement", "I turned east into the library")
Two location names.
Example：
1.Observation: I walked south from the main hall into a quiet hallway.
Reasoning: I walked south from the main hall into the hallway.
Label: HORIZONTAL

⬜ ⬜ ⬜  
⬜ 🚩 ⬜  
⬜ 🏃 ⬜  


Previous location: main hall
Current location: hallway

2.Observation: I climbed a staircase from the hallway to the balcony above.
Reasoning: I went up from the hallway to the balcony.
Label: VERTICAL

⬜ 🏃 ⬜  
⬜ 🚩 ⬜  
⬜ ⬜ ⬜  


Previous location: hallway
Current location: balcony

3.Observation: I remained in the garden, enjoying the scenery.
Reasoning: I stayed in the same place, the garden.
Label: STAY

⬜ ⬜ ⬜  
⬜ 🏃 ⬜  
⬜ ⬜ ⬜  

Previous location: garden
Current location: garden


Notice: Always set previous location 🚩 in the middle of map.

    """
    history_file_name_template="history_"+str(path_name.split('/')[-1])+".json"
    valid_name_template="valid_response_"+str(path_name.split('/')[-1])+".json"

    messages = [
        message_template('system',sys_prompt),

    ]

    for each_walk in tqdm(walkthrough):
        print_colored(str(each_walk),'blue')
        messages.append(message_template('user', each_walk))
        valid_map_list.append(message_template('user', each_walk))

        response = chat_single(messages,verbose=True).lower()

        messages.append(message_template('assistant', response))
        # 循环直到 response 中的 location 合法
        while True:
            validation_result = validate_locations(response, location_path)
            if not validation_result:
                break  # 验证通过，跳出循环
            print_colored(str(validation_result), 'red')

            messages.append(message_template('user', validation_result))
            response = chat_single(messages,verbose=True).lower()
            messages.append(message_template('assistant', response))
        valid_map_list.append(message_template('assistant', response))
    save_list_to_json(messages,history_file_name_template)
    save_list_to_json(valid_map_list,valid_name_template)

    # messages.append(message_template('user', str(walkthrough)))
    # response = chat_single(messages)
    # print(response)
from pathlib import Path
path_name='D:/mango/data/night'
DATA_DIR = Path(path_name)
#
#
# file_name_template = "history" + ".json"
# history_file_name = get_available_filename(file_name_template)
# print(history_file_name)
eval_model(path_name,walkthrough=read_walkthrough(DATA_DIR / "night.walkthrough",70))