import numpy as np

def compute_success(final_pos, goal_pos, threshold=3.0):
    dist = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
    return float(dist < threshold), round(float(dist), 3)

def compute_spl(success, shortest_path, actual_path):
    if shortest_path == 0:
        return 1.0 if success else 0.0
    return success * (shortest_path / max(shortest_path, actual_path))

# Example episode
goal     = [3.0, 0.0, 4.0]
final    = [2.8, 0.0, 3.9]
shortest = 5.0
actual   = 8.3

sr, ne = compute_success(final, goal)
spl    = compute_spl(sr, shortest, actual)

print("=" * 40)
print("        EVALUATION METRICS")
print("=" * 40)
print(f"Navigation Error (NE) : {ne} m")
print(f"Success Rate (SR)     : {sr}  (1=success, 0=fail)")
print(f"SPL                   : {spl:.3f}  (1.0=perfect)")
print("=" * 40)
print()
print("DEFINITIONS:")
print("NE  = final distance to goal in metres (lower is better)")
print("SR  = 1 if agent stops within 3m of goal, else 0")
print("SPL = SR x (shortest_path / actual_path_taken)")
print("      rewards success AND penalizes inefficient paths")
