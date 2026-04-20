import habitat_sim
import numpy as np
import torch
import cv2
import math
import sys
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from task3.model import VLNModel, ACTION_NAMES

MODEL_PATH = os.path.abspath(
    os.path.join(PROJECT_ROOT, "task3", "best_model_honest.pth")
)

SCENE = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        "habitat-lab",
        "data",
        "scene_datasets",
        "habitat-test-scenes",
        "van-gogh-room.glb"
    )
)

SUCCESS_THRESHOLD = 3.0   # ✅ standard VLN threshold
MAX_STEPS = 80

TEST_EPISODES = [
    {"instruction": "Walk forward"},
    {"instruction": "Turn left and go forward"},
    {"instruction": "Turn right"},
    {"instruction": "Move ahead and stop"},
]


def euclidean(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color_sensor"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [256, 256]
    rgb.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]

    # ✅ DEFINE ACTION SPACE
    agent_cfg.action_space = {
        "MOVE_FORWARD": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "TURN_LEFT": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "TURN_RIGHT": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "STOP": habitat_sim.agent.ActionSpec(
            "stop", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
    }

    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    )


def run_episode(sim, model, episode):
    agent = sim.initialize_agent(0)

    # ✅ VALID START
    start = sim.pathfinder.get_random_navigable_point()
    state = habitat_sim.AgentState()
    state.position = start
    agent.set_state(state)

    start_pos = np.array(start).tolist()
    goal = np.array(sim.pathfinder.get_random_navigable_point()).tolist()

    instruction = episode["instruction"]

    hidden = None
    prev_pos = start_pos[:]
    total_move = 0.0

    for step in range(MAX_STEPS):
        obs = sim.get_sensor_observations()

        rgb = np.array(obs["color_sensor"])

        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]

        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)

        curr = agent.get_state().position.tolist()
        dist = euclidean(curr, goal)

        moved = euclidean(prev_pos, curr)
        total_move += moved
        prev_pos = curr[:]

        # ✅ SUCCESS CONDITION
        if dist < SUCCESS_THRESHOLD and total_move > 1.0:
            return True, dist

        rgb_t = torch.tensor(rgb).permute(2, 0, 1).float()

        action, hidden = model.predict_action(
            rgb_t, instruction, hidden
        )

        action_name = ACTION_NAMES[action]

        if action_name == "STOP":
            success = (dist < SUCCESS_THRESHOLD) and (total_move > 1.0)
            return success, dist

        sim.step(action_name)

    final = np.array(agent.get_state().position).tolist()
    dist = euclidean(final, goal)

    success = (dist < SUCCESS_THRESHOLD) and (total_move > 1.0)

    return success, dist


if __name__ == "__main__":
    device = torch.device("cpu")

    print("Loading model...")
    model = VLNModel(feature_dim=512, num_actions=4).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sim = make_sim()

    successes = 0
    distances = []

    print("\nRunning Task 4 TEST...")

    for i, ep in enumerate(TEST_EPISODES):
        success, dist = run_episode(sim, model, ep)

        distances.append(dist)
        if success:
            successes += 1

        print(f"Ep {i+1}: {ep['instruction']}")
        print(f"   Dist={dist:.2f} | {'SUCCESS' if success else 'FAIL'}")

    sim.close()

    sr = successes / len(TEST_EPISODES)
    ne = sum(distances) / len(distances)

    print("\n==============================")
    print("TASK 4 TEST RESULTS")
    print("==============================")
    print(f"SR: {sr:.3f}")
    print(f"NE: {ne:.3f}")