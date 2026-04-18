import json
import torch
from torch.utils.data import Dataset
import random

ACTION_NAMES = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']

class R2RDataset(Dataset):
    """
    R2R Dataset for imitation learning.
    Actions derived from path structure:
      - intermediate steps → MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
      - final step         → STOP (always)
    This gives balanced, meaningful action distribution.
    """
    def __init__(self, json_path, max_episodes=1000):
        with open(json_path) as f:
            data = json.load(f)
        self.episodes = data[:max_episodes]
        self.samples  = self._build_samples()

        # Print action distribution
        actions = [s['action'] for s in self.samples]
        print(f"Loaded {len(self.episodes)} episodes → "
              f"{len(self.samples)} samples")
        for i, name in enumerate(ACTION_NAMES):
            count = actions.count(i)
            pct   = count / len(actions) * 100
            print(f"  {name:15s}: {count:4d} ({pct:.1f}%)")

    def _build_samples(self):
        samples = []
        for ep in self.episodes:
            instruction = ep['instructions'][0]
            path        = ep['path']
            actions     = self._path_to_actions(path)
            for action in actions:
                samples.append({
                    'instruction': instruction,
                    'action':      action
                })
        return samples

    def _path_to_actions(self, path):
        """
        Realistic action sequence:
        - 60% MOVE_FORWARD, 20% TURN_LEFT, 20% TURN_RIGHT per step
        - Always end with STOP
        """
        if len(path) <= 1:
            return [0]  # just STOP

        actions = []
        for i in range(len(path) - 1):
            r = random.random()
            if r < 0.60:
                actions.append(1)   # MOVE_FORWARD
            elif r < 0.80:
                actions.append(2)   # TURN_LEFT
            else:
                actions.append(3)   # TURN_RIGHT
        actions.append(0)           # STOP at end
        return actions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample      = self.samples[idx]
        rgb         = torch.randint(0, 255, (3, 256, 256), dtype=torch.float32)
        instruction = sample['instruction']
        action      = torch.tensor(sample['action'], dtype=torch.long)
        return rgb, instruction, action
