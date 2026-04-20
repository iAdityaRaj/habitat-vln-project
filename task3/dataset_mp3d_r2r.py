import habitat_sim, torch, numpy as np
import json, random, os
from torch.utils.data import Dataset

ACTION_NAMES = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
MP3D_SCENE   = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

VALID_POSITIONS = [
    [-1.5, 0.0, 12.0], [-2.0, 0.0, 10.0], [-1.0, 0.0, 11.0],
    [-2.5, 0.0, 12.5], [-1.5, 0.0, 13.0], [-3.0, 0.0, 11.5],
    [-1.0, 0.0, 10.5], [-2.0, 0.0, 13.0],
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

def load_instructions_by_action(json_path, max_eps=500):
    """
    Load R2R instructions and bucket them by action keyword.
    Returns dict: {action_id: [list of instructions]}
    """
    with open(json_path) as f:
        data = json.load(f)

    buckets = {0: [], 1: [], 2: [], 3: []}
    for ep in data[:max_eps]:
        for inst in ep['instructions']:
            inst = inst.strip()
            low  = inst.lower()
            if any(w in low for w in ['stop', 'halt', 'wait',
                                       'reached', 'destination']):
                buckets[0].append(inst)
            elif any(w in low for w in ['turn left', 'go left',
                                         'bear left', 'left turn']):
                buckets[2].append(inst)
            elif any(w in low for w in ['turn right', 'go right',
                                          'bear right', 'right turn']):
                buckets[3].append(inst)
            else:
                buckets[1].append(inst)

    for action, insts in buckets.items():
        print(f"  {ACTION_NAMES[action]:15s}: {len(insts)} instructions")

    return buckets


class R2RMP3DDataset(Dataset):
    """
    Balanced dataset: 25% each action.
    Uses real R2R instructions bucketed by action keyword.
    Renders real MP3D RGB at diverse positions.
    """
    def __init__(self, r2r_path, split='train',
                 samples_per_action=80, seed=42):
        random.seed(seed + (0 if split=='train' else 1))
        np.random.seed(seed)

        print(f"\nLoading R2R instructions for {split}...")
        all_buckets = load_instructions_by_action(r2r_path)

        # Split each bucket 80/20
        self.buckets = {}
        for action, insts in all_buckets.items():
            random.shuffle(insts)
            n = int(0.8 * len(insts))
            if split == 'train':
                self.buckets[action] = insts[:n] if insts else []
            else:
                self.buckets[action] = insts[n:] if insts else []
            # If bucket empty, use generic fallback
            if not self.buckets[action]:
                self.buckets[action] = [ACTION_NAMES[action] + " action"]

        self.samples  = []
        self.split    = split
        self.n_per_action = samples_per_action
        self._collect()

        # Print distribution
        actions = [s['action'] for s in self.samples]
        print(f"{split}: {len(self.samples)} samples")
        for i, name in enumerate(ACTION_NAMES):
            c = actions.count(i)
            print(f"  {name:15s}: {c:4d} ({c/len(actions)*100:.1f}%)")

    def _collect(self):
        print(f"Rendering real MP3D RGB ({self.split})...")
        sim   = make_sim()
        agent = sim.initialize_agent(0)
        valid = list(agent.agent_config.action_space.keys())

        # Collect equal samples per action — BALANCED
        for action_id in range(4):
            insts = self.buckets[action_id]
            for i in range(self.n_per_action):
                # Random position
                pos   = random.choice(VALID_POSITIONS)
                state = habitat_sim.AgentState()
                state.position = np.array([
                    pos[0] + random.uniform(-0.3, 0.3),
                    pos[1],
                    pos[2] + random.uniform(-0.3, 0.3)
                ])
                agent.set_state(state)

                # Random steps for diverse views
                for _ in range(random.randint(0, 10)):
                    sim.step(random.choice(valid))

                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][:,:,:3].copy()

                instruction = random.choice(insts)
                self.samples.append({
                    'rgb':         rgb,
                    'instruction': instruction,
                    'action':      action_id,
                })

        sim.close()
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        rgb = torch.tensor(s['rgb']).permute(2,0,1).float()
        return rgb, s['instruction'], torch.tensor(s['action'], dtype=torch.long)
