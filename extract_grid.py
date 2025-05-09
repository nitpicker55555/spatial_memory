import re


def parse_table_string(table_str):
    """
    Parse a table-formatted string and extract structured data with location,
    coordinates, reasoning, and links (with direction and action).

    Returns:
        dict with:
            - nodes: list of dicts with 'name', 'coordinates'
            - links: list of dicts with 'source', 'target', 'action', 'reasoning'
    """
    lines = table_str.strip().splitlines()
    data_lines = [line for line in lines if
                  '|' in line and not re.match(r'^\s*\|[-\s|]+\|\s*$', line)]

    nodes = []
    links = []
    reasoning = []

    prev_name = None

    for line in data_lines[1:]:  # skip header
        parts = [part.strip() for part in line.strip('|').split('|')]
        if len(parts) >= 5:
            name = parts[0]
            try:
                x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                reasoning_str = parts[4]

                node = {
                    'name': name,
                    'coordinates': (x, y, z),

                }
                nodes.append(node)
                reasoning.append({
                    "reasoning":reasoning_str
                })

                # If previous node exists, build a link
                if prev_name:
                    # Try to extract direction or action from the reasoning string
                    match = re.search(
                        r'\b(moved|walked|went|followed|turned)\s+([a-z\s]+?)\b(from|through|to|out|into)',
                        reasoning, re.IGNORECASE)
                    action = match.group(
                        2).strip().lower() if match else 'unknown'
                    links.append({
                        'source': prev_name,
                        'target': name,
                        'action': action,

                    })

                prev_name = name

            except ValueError:
                continue  # skip invalid coordinate rows

    return {
        'nodes': nodes,
        'links': links,
        "reasoning":reasoning
    }


# 示例表格
table_string = '''
| Location Name           | X  | Y  | Z  | Reasoning                                                                 |
|-------------------------|----|----|----|---------------------------------------------------------------------------|
| Computer Site           | 0  | 0  | 0  | Initial position                                                          |
| Hall Outside Computer Site | 1  | 1  | 0  | I moved northeast from the Computer Site                                 |
| Hall                    | 1  | 0  | 0  | I walked south from the Hall Outside Computer Site                        |
| Hall Outside Elevator   | 2  | 0  | 0  | I moved east from the Hall                                                |
| Stairwell (Third Floor) | 0  | 1  | 0  | I moved west from the Hall Outside Computer Site                          |
| Stairwell (Second Floor)| 0  | 1  | -1 | I went down from the Stairwell (Third Floor)                              |
| Stairwell (First Floor) | 0  | 1  | -2 | I went down from the Stairwell (Second Floor)                             |
| Hall (First Floor)      | 1  | 1  | -2 | I moved east from the Stairwell (First Floor)                             |
| Hall (Middle)           | 1  | 0  | -2 | I walked south from the Hall (First Floor)                                |
| Hall Outside Elevator (First Floor) | 2  | 0  | -2 | I moved east from the Hall (Middle)                                   |
| Janitor's Closet        | 2  | -1 | -2 | I moved south from the Hall Outside Elevator (First Floor)                |
| Outside Physics Office  | 1  | 1  | -1 | I moved east from the Stairwell (Second Floor)                            |
| Maze of Twisty Passages | 1  | 0  | -3 | I went down through the panel from the Hall (Middle)                      |
| Gnome's Lair            | 0  | 0  | -3 | I followed the mouse out of the maze to the Gnome's Lair                  |
'''

# # 运行解析
graph_data = parse_table_string(table_string)

# 打印结果
for node in graph_data['nodes']:
    print(node)

print("\nLinks:")
for link in graph_data['links']:
    print(link)
