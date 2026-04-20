"""
Real RGB dataset using actual MP3D scene (17DRP5sb8fy).
Renders real Habitat observations for training.
"""

import habitat_sim
import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset

ACTION_NAMES = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']

MP3D_SCENE = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

# Real R2R-style instructions paired with actions
INSTRUCTION_ACTION_PAIRS = [
    ("Walk forward down the hallway and stop",          1),
    ("Turn left and move to the room entrance",         2),
    ("Go straight ahead and stop near the far wall",    1),
    ("Move forward three steps then stop",              1),
    ("Turn right and walk to the corner",               3),
    ("Navigate straight to the end of the corridor",    1),
    ("Go forward and turn left at the junction",        2),
    ("Walk to the doorway and stop",                    1),
    ("Move ahead to the center of the room",            1),
    ("Go straight forward to the window",               1),
    ("Turn left and walk through the room",             2),
    ("Move forward to the far wall",                    1),
    ("Turn right at the junction and stop",             3),
    ("Walk forward and stop at the desk",               1),
    ("Stop here you have reached the destination",      0),
    ("Turn left now and proceed forward",               2),
    ("Turn right and continue down the hall",           3),
    ("You are at the goal please stop",                 0),
    ("Walk into the living room and stop",              1),
    ("Turn left towards the sofa and stop",             2),
    ("Go right past the window",                        3),
    ("Stop when you reach the couch",                   0),
    ("Move forward towards the entrance",               1),
    ("Turn right and face the door",                    3),
    ("Stop at the end of the corridor",                 0),
]

def make_sim():
    sim_cfg          = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = MP3D_SCENE
    rgb              = habitat_sim.CameraSensorSpec()
    rgb.uuid         = "color_sensor"
    rgb.sensor_type  = habitat_sim.SensorType.COLOR
    rgb.resolution   = [256, 256]
    rgb.position     = [0.0, 1.5, 0.0]
    agent_cfg        = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    )


class MP3DDataset(Dataset):
    """
    Real RGB training data from MP3D scene 17DRP5sb8fy.
    Agent navigates around and captures RGB at diverse positions.
    """
    def __init__(self, num_samples=800, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.samples = []
        self._collect(num_samples)

    def _collect(self, num_samples):
        print("Loading real MP3D scene...")
        sim   = make_sim()
        agent = sim.initialize_agent(0)
        valid = list(agent.agent_config.action_space.keys())

        print(f"Collecting {num_samples} real RGB observations...")

        for i in range(num_samples):
            # Reset to a random position by taking random steps from start
            state          = habitat_sim.AgentState()
            state.position = np.array([-1.5, 0.0, 12.0])  # valid start in 17DRP5sb8fy
            agent.set_state(state)

            # Take random steps to get diverse viewpoints
            n_steps = random.randint(0, 15)
            for _ in range(n_steps):
                sim.step(random.choice(valid))

            # Get real RGB
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][:,:,:3].copy()

            # Pair with instruction and action
            instruction, action = random.choice(INSTRUCTION_ACTION_PAIRS)

            self.samples.append({
                'rgb':         rgb,
                'instruction': instruction,
                'action':      action,
            })

            if (i+1) % 100 == 0:
                print(f"  Collected {i+1}/{num_samples} samples")

        sim.close()

        # Print distribution
        actions = [s['action'] for s in self.samples]
        print(f"\nDataset ready: {len(self.samples)} samples")
        for i, name in enumerate(ACTION_NAMES):
            c = actions.count(i)
            print(f"  {name:15s}: {c:4d} ({c/len(actions)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        rgb = torch.tensor(s['rgb']).permute(2,0,1).float()
        return rgb, s['instruction'], torch.tensor(s['action'], dtype=torch.long)
