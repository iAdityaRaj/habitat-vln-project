import os
import subprocess

SCENES = [
    "17DRP5sb8fy",  # seen
    "1LXtFkjw3qL",  # unseen
]

BASE_PATH = "../habitat-lab/data/scene_datasets/mp3d_example"

def run(scene_id):
    scene_path = f"{BASE_PATH}/{scene_id}/{scene_id}.glb"

    print(f"\nRunning scene: {scene_id}")

    os.environ["SCENE_PATH"] = scene_path

    subprocess.run([
        "python", "../evaluate_aligned.py"
    ])

if __name__ == "__main__":
    for scene in SCENES:
        run(scene)