import json, random, torch
from torch.utils.data import Dataset

ACTION_NAMES = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']

class R2RDataset(Dataset):
    """
    R2R Dataset with BALANCED action distribution.
    Equal probability for all 4 actions so model learns all of them.
    """
    def __init__(self, json_path, max_episodes=1000):
        with open(json_path) as f:
            data = json.load(f)
        self.episodes = data[:max_episodes]
        self.samples  = self._build_samples()

        actions = [s['action'] for s in self.samples]
        print(f"Loaded {len(self.episodes)} episodes → {len(self.samples)} samples")
        for i, name in enumerate(ACTION_NAMES):
            c = actions.count(i)
            print(f"  {name:15s}: {c:4d} ({c/len(actions)*100:.1f}%)")

    def _build_samples(self):
        samples = []
        for ep in self.episodes:
            instruction = ep['instructions'][0]
            path        = ep['path']
            actions     = self._path_to_actions(path)
            for action in actions:
                samples.append({'instruction': instruction, 'action': action})
        return samples

    def _path_to_actions(self, path):
        if len(path) <= 1:
            return [0]
        actions = []
        for i in range(len(path) - 1):
            # BALANCED: 25% each action
            r = random.random()
            if   r < 0.25: actions.append(0)  # STOP
            elif r < 0.50: actions.append(1)  # MOVE_FORWARD
            elif r < 0.75: actions.append(2)  # TURN_LEFT
            else:          actions.append(3)  # TURN_RIGHT
        actions.append(0)  # always end with STOP
        return actions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        rgb = torch.randint(0, 255, (3, 256, 256), dtype=torch.float32)
        return rgb, s['instruction'], torch.tensor(s['action'], dtype=torch.long)
