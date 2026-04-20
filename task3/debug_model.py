"""Check what the model actually predicts for each instruction type."""
import torch
from model import VLNModel, ACTION_NAMES

model = VLNModel(feature_dim=512, num_actions=4)
ckpt  = torch.load("best_model_honest.pth", map_location="cpu")
model.load_state_dict(ckpt['model_state'])
model.eval()

# Test with dummy RGB + each instruction type
dummy_rgb = torch.zeros(3, 256, 256)  # black image

test_instructions = [
    # STOP
    ("Stop here", 0),
    ("You have arrived at your destination", 0),
    ("Halt now", 0),
    # FORWARD
    ("Walk forward", 1),
    ("Move ahead", 1),
    ("Head in the forward direction", 1),
    # LEFT
    ("Turn left", 2),
    ("Make a left", 2),
    ("Veer to the left side", 2),
    # RIGHT
    ("Turn right", 3),
    ("Make a right", 3),
    ("Veer to the right side", 3),
    # Evaluation instructions
    ("Walk forward down the hallway and stop at the end", 1),
    ("Turn left and move to the room entrance", 2),
    ("Go straight ahead and stop near the far wall", 1),
    ("Navigate straight to the end of the hallway", 1),
    ("Go forward and turn left at the junction", 2),
]

print("=" * 60)
print("  MODEL PREDICTION TEST")
print("=" * 60)
print(f"{'Instruction':<45} {'Expected':>8} {'Predicted':>10}")
print("-" * 60)

correct = 0
for instruction, expected in test_instructions:
    action_id, _ = model.predict_action(dummy_rgb, instruction)
    pred_name    = ACTION_NAMES[action_id]
    exp_name     = ACTION_NAMES[expected]
    match        = "✓" if action_id == expected else "✗"
    correct     += (action_id == expected)
    print(f"{instruction[:44]:<45} {exp_name:>8} {pred_name:>10} {match}")

print("-" * 60)
print(f"Accuracy: {correct}/{len(test_instructions)} = {correct/len(test_instructions):.1%}")
