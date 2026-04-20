import json, random

def instruction_to_action(instruction):
    inst = instruction.lower()
    if any(w in inst for w in ['stop', 'halt', 'wait', 'stay',
                                'reached', 'destination', 'end']):
        return 0
    if any(w in inst for w in ['turn left', 'go left', 'left turn',
                                'rotate left', 'face left', 'bear left']):
        return 2
    if any(w in inst for w in ['turn right', 'go right', 'right turn',
                                'rotate right', 'face right', 'bear right']):
        return 3
    return 1

with open("data/R2R_train.json") as f:
    data = json.load(f)

instructions = []
for ep in data[:500]:
    for inst in ep['instructions']:
        instructions.append(inst.strip())
instructions = list(set(instructions))

actions = [instruction_to_action(i) for i in instructions]
names   = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
print("Action distribution from R2R instructions:")
for i, name in enumerate(names):
    c = actions.count(i)
    print(f"  {name:15s}: {c:4d} ({c/len(actions)*100:.1f}%)")
