"""
Honest training with proper train/val split.
Train on simple instruction style.
Validate on DIFFERENT instruction style the model has never seen.
"""
import habitat_sim, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from model import VLNModel, ACTION_NAMES
from metrics import compute_metrics

SCENE = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

NAVIGABLE = [
    [-2.026, 0.072, -2.482], [-5.084, 0.072, -1.585],
    [-0.785, 0.072,  0.656], [-1.118, 0.072, -1.522],
    [-0.608, 0.072,  1.880], [ 0.085, 0.072,  0.830],
    [-3.748, 0.072, -2.091], [-7.510, 0.072, -0.806],
    [ 0.959, 0.072,  0.450], [-6.150, 0.072,  0.415],
    [ 1.536, 0.072, -0.386], [ 2.485, 0.072,  0.600],
    [-2.647, 0.072, -3.917], [-3.610, 0.072, -3.294],
    [-6.521, 0.072, -0.822],
]

# TRAIN instructions — simple imperative style
TRAIN_INSTRUCTIONS = {
    0: ["Stop here", "Halt now", "Stop at this point",
        "Stay here", "Do not move"],
    1: ["Walk forward", "Move ahead", "Go straight",
        "Proceed forward", "Continue straight"],
    2: ["Turn left", "Go left", "Rotate left",
        "Bear left", "Face left"],
    3: ["Turn right", "Go right", "Rotate right",
        "Bear right", "Face right"],
}

# VAL instructions — different phrasing (model never sees these during train)
VAL_INSTRUCTIONS = {
    0: ["You have arrived at your destination",
        "This is where you need to be",
        "End your journey here"],
    1: ["Head in the forward direction",
        "Advance toward what is ahead",
        "Keep going straight"],
    2: ["Make a left",
        "Veer to the left side",
        "Swing left"],
    3: ["Make a right",
        "Veer to the right side",
        "Swing right"],
}

def make_sim():
    sim_cfg          = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
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

class VLNDataset(Dataset):
    def __init__(self, instruction_set, n_per_action=120, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.samples = []
        self._collect(instruction_set, n_per_action)

    def _collect(self, instruction_set, n_per_action):
        sim   = make_sim()
        agent = sim.initialize_agent(0)
        valid = list(agent.agent_config.action_space.keys())

        for action_id in range(4):
            for _ in range(n_per_action):
                pos   = random.choice(NAVIGABLE)
                state = habitat_sim.AgentState()
                state.position = np.array([
                    pos[0]+random.uniform(-0.2, 0.2),
                    pos[1],
                    pos[2]+random.uniform(-0.2, 0.2)
                ])
                agent.set_state(state)

                for _ in range(random.randint(0, 8)):
                    sim.step(random.choice(valid))

                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][:,:,:3].copy()
                instruction = random.choice(instruction_set[action_id])

                self.samples.append({
                    'rgb': rgb,
                    'instruction': instruction,
                    'action': action_id,
                })

        sim.close()
        random.shuffle(self.samples)

        actions = [s['action'] for s in self.samples]
        print(f"  {len(self.samples)} samples | ", end="")
        for i, n in enumerate(ACTION_NAMES):
            print(f"{n}={actions.count(i)}", end=" ")
        print()

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s['rgb']).permute(2,0,1).float(),
                s['instruction'],
                torch.tensor(s['action'], dtype=torch.long))


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0; all_p, all_t = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for rgb, instr, actions in tqdm(
                loader, desc="Train" if train else "Val  ", leave=False):
            rgb = rgb.to(device); actions = actions.to(device)
            if train: optimizer.zero_grad()
            logits, _ = model(rgb, list(instr), hidden_state=None)
            loss = criterion(logits, actions)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            total_loss += loss.item()
            all_p.append(torch.argmax(logits,1).cpu())
            all_t.append(actions.cpu())
    all_p = torch.cat(all_p); all_t = torch.cat(all_t)
    acc   = (all_p==all_t).float().mean().item()
    m     = compute_metrics(all_p, all_t)
    return total_loss/len(loader), acc, m


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    print("Building TRAIN dataset (simple instructions)...")
    train_ds = VLNDataset(TRAIN_INSTRUCTIONS, n_per_action=120)
    print("Building VAL dataset (UNSEEN instruction style)...")
    val_ds   = VLNDataset(VAL_INSTRUCTIONS,   n_per_action=40, seed=99)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    model     = VLNModel(feature_dim=512, num_actions=4).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history = {'train_loss':[], 'val_loss':[],
               'train_acc':[], 'val_acc':[], 'sr':[], 'spl':[], 'ne':[]}
    best_sr = 0.0
    NUM_EPOCHS = 25

    print("\n" + "="*55)
    print("  HONEST TRAINING — UNSEEN VAL INSTRUCTIONS")
    print("  Val accuracy reflects real generalization")
    print("="*55)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        tl, ta, _ = run_epoch(model, train_loader, optimizer,
                               criterion, device, train=True)
        vl, va, m = run_epoch(model, val_loader,   optimizer,
                               criterion, device, train=False)
        scheduler.step()

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        history['sr'].append(m['sr'])
        history['spl'].append(m['spl'])
        history['ne'].append(m['ne'])

        print(f"  Train → Loss:{tl:.4f} Acc:{ta:.3f}")
        print(f"  Val   → Loss:{vl:.4f} Acc:{va:.3f}  ← honest")
        print(f"  SR={m['sr']:.3f} SPL={m['spl']:.3f} NE={m['ne']:.3f}")
        print(f"  Per-action: ", end="")
        for k,v in m['action_acc'].items():
            print(f"{k}={v:.2f}", end=" ")
        print()

        if m['sr'] >= best_sr:
            best_sr = m['sr']
            torch.save({'model_state': model.state_dict(),
                        'sr': m['sr'], 'spl': m['spl']},
                       "best_model_honest.pth")
            print(f"  ✅ Saved! SR={m['sr']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    epochs = range(1, NUM_EPOCHS+1)
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val (unseen)')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val (unseen)')
    axes[1].set_title('Accuracy'); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history['sr'],  'g-o', label='SR')
    axes[2].plot(epochs, history['spl'], 'm-o', label='SPL')
    axes[2].plot(epochs, history['ne'],  'c-o', label='NE')
    axes[2].set_title('SR / SPL / NE'); axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.suptitle(
        'CLIP-VLN — Honest Training\n'
        'Val uses UNSEEN instruction phrasing', fontsize=12)
    plt.tight_layout()
    plt.savefig("learning_curves_honest.png", dpi=150)
    print(f"\nBest Val SR : {best_sr:.4f}")
    print(f"Model       : best_model_honest.pth")
    print(f"Curves      : learning_curves_honest.png")
    print("\nopen learning_curves_honest.png")
