from chat_py import chat_single, message_template
from eval import read_walkthrough



def model(walkthrough:list):
    sys_prompt="""
You are exploring a world described in natural language, step by step.

Your task is to **construct a 3D mind graph** where each room or location is assigned a unique coordinate (X, Y, Z), starting from the origin (0, 0, 0).

Each observation describes your current location. Based on the direction you just moved and the textual description, you must:
- Assign a coordinate to the current location relative to your previous one
- Fill in a short explanation of your spatial reasoning (e.g., “I climbed up from the previous room” or “this place is east of the kitchen”)

All positions must be consistent. That means:
- The same location name must always have the same coordinate
- If a location is revisited, its coordinate must match the earlier assignment
- Directions must obey spatial rules (e.g., “north” means Y+1, “down” means Z−1)

You may also infer relative positions based on natural language clues (e.g., “back of the house” implies “south of house”)

### Coordinate conventions:
- north → Y + 1
- south → Y − 1
- east → X + 1
- west → X − 1
- up → Z + 1
- down → Z − 1
- northeast → X + 1, Y + 1
- southwest → X − 1, Y − 1
(... and so on for diagonals)
Actions not listed in there can be treat as same action previously

Output the current state of your mind graph as a table:

| Location Name     | X  | Y  | Z | Action             | Reasoning                              |
| ----------------- | -- | -- | - | ------------------ | -------------------------------------- |
| Starting Location | 0  | 0  | 0 | —                  | Initial position                       |
| Hallway           | 0  | -1 | 0 | walked south       | I walked south from the starting point |
| Balcony           | 0  | -1 | 1 | went up            | I went up from the hallway             |
| Garden            | 1  | 0  | 0 | turned east        | I turned east and exited to the garden |
| Temple            | -1 | 1  | 0 | inferred northwest | Described as northwest of the garden   |

Update this table **after each observation**, one row at a time.


    """
    messages = [
        message_template('system',sys_prompt),

    ]

    # for each_walk in walkthrough:
    #     messages.append(        message_template('user', each_walk))
    #     response=chat_single(messages)
    #     print(response)
    messages.append(message_template('user', str(walkthrough)))
    response = chat_single(messages)
    print(response)
from pathlib import Path
DATA_DIR = Path(r"D:/mango/data/night")


model(walkthrough=read_walkthrough(DATA_DIR / "night.walkthrough",70))