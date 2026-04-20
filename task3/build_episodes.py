"""Find good episode pairs where goal is 3-4m away."""
import math

NAVIGABLE = [
    [-2.026, 0.072, -2.482],
    [-5.084, 0.072, -1.585],
    [-0.785, 0.072,  0.656],
    [-1.118, 0.072, -1.522],
    [-0.608, 0.072,  1.880],
    [ 0.085, 0.072,  0.830],
    [-3.748, 0.072, -2.091],
    [-7.510, 0.072, -0.806],
    [ 0.959, 0.072,  0.450],
    [-6.150, 0.072,  0.415],
    [ 1.536, 0.072, -0.386],
    [ 2.485, 0.072,  0.600],
    [-2.647, 0.072, -3.917],
    [-3.610, 0.072, -3.294],
    [-6.521, 0.072, -0.822],
]

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

print("Finding pairs 2.5-4.0m apart:")
pairs = []
for i, s in enumerate(NAVIGABLE):
    for j, g in enumerate(NAVIGABLE):
        if i == j: continue
        d = dist(s, g)
        if 2.5 <= d <= 4.0:
            pairs.append((i, j, d, s, g))
            print(f"  {i}→{j}: {d:.2f}m | {s} → {g}")

print(f"\nFound {len(pairs)} valid pairs")
