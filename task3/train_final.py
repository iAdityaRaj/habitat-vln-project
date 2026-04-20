"""
Final proper training:
- Explicitly teach turning by rotating agent before capture
- Each instruction ALWAYS corresponds to what agent is about to do
- Diverse positions and headings
"""
import habitat_sim, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random, json
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

# Action-specific instruction templates
ACTION_INSTRUCTIONS = {
    0: [  # STOP
        "Stop here you have reached the destination",
        "You are at the goal please stop now",
        "Halt here this is your destination",
        "Stop at this location you have arrived",
        "This is the end of your route stop here",
    ],
    1: [  # MOVE_FORWARD
        "Walk forward down the hallway",
        "Move ahead toward the room",
        "Go straight forward to the wall",
        "Continue walking forward",
        "Proceed straight ahead",
        "Walk forward toward the entrance",
        "Move forward down the corridor",
        "Go straight to the end",
    ],
    2: [  # TURN_LEFT
        "Turn left at the junction",
        "Go left toward the door",
        "Turn left and proceed",
        "Bear left at the corridor",
        "Rotate left to face the room",
        "Take a left turn here",
        "Turn to your left now",
        "Go left past the window",
    ],
    3: [  # TURN_RIGHT
        "Turn right at the corner",
        "Go right toward the exit",
        "Turn right and continue",
        "Bear right at the hallway",
        "Rotate right to face the door",
        "Take a right turn here",
        "Turn to your right now",
        "Go right past the stairs",
    ],
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

class ActionAlignedDataset(Dataset):
    """
    Key insight: capture RGB BEFORE taking the action.
    The label is the NEXT action the agent will take.
    This teaches the model to predict the right action
    given the current visual context.
    """
    def __init__(self, n_per_action=150, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.samples = []
        self._collect(n_per_action)

    def _collect(self, n_per_action):
        print("Collecting action-aligned RGB samples...")
        sim   = make_sim()
        agent = sim.initialize_agent(0)
        valid = list(agent.agent_config.action_space.keys())

        for action_id in range(4):
            print(f"  Collecting {n_per_action} samples for {ACTION_NAMES[action_id]}...")
            count = 0
            attempts = 0
            while count < n_per_action and attempts < n_per_action * 3:
                attempts += 1
                # Random navigable position
                pos   = random.choice(NAVIGABLE)
                state = habitat_sim.AgentState()
                state.position = np.array([
                    pos[0] + random.uniform(-0.2, 0.2),
                    pos[1],
                    pos[2] + random.uniform(-0.2, 0.2)
                ])
                # Random heading
                angle = random.uniform(0, 2*np.pi)
                state.rotation = np.quaternion(
                    np.cos(angle/2), 0, np.sin(angle/2), 0
                )
                agent.set_state(state)

                # Take a few random steps for diversity
                for _ in range(random.randint(0, 5)):
                    sim.step(random.choice(valid))

                # Capture RGB BEFORE action
                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][:,:,:3].copy()

                # Instruction matches the action
                instruction = random.choice(ACTION_INSTRUCTIONS[action_id])

                # Verify action is executable (agent not stuck)
                pos_before = agent.get_state().position.tolist()
                if action_id in [1,2,3]:
                    action_name = valid[action_id-1] if action_id <= len(valid) else valid[0]
                    if action_id == 1: action_name = 'move_forward'
                    if action_id == 2: action_name = 'turn_left'
                    if action_id == 3: action_name = 'turn_right'
                    if action_name in valid:
                        sim.step(action_name)

                self.samples.append({
                    'rgb': rgb,
                    'instruction': instruction,
                    'action': action_id,
                })
                count += 1

        sim.close()
        random.shuffle(self.samples)

        # Print distribution
        actions = [s['action'] for s in self.samples]
        print(f"\nDataset: {len(self.samples)} samples")
        for i, name in enumerate(ACTION_NAMES):
            c = actions.count(i)
            print(f"  {name:15s}: {c:4d} ({c/len(actions)*100:.1f}%)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s['rgb']).permute(2,0,1).float(),
                s['instruction'],
                torch.tensor(s['action'], dtype=torch.long))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for rgb, instr, actions in tqdm(loader, desc="Train", leave=False):
        rgb = rgb.to(device); actions = actions.to(device)
        optimizer.zero_grad()
        logits, _ = model(rgb, list(instr), hidden_state=None)
        loss = criterion(logits, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        correct    += (torch.argmax(logits,1)==actions).sum().item()
        total      += actions.size(0)
    return total_loss/len(loader), correct/total

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0; all_p, all_t = [], []
    with torch.no_grad():
        for rgb, instr, actions in tqdm(loader, desc="Val", leave=False):
            rgb = rgb.to(device); actions = actions.to(device)
            logits, _ = model(rgb, list(instr), hidden_state=None)
            total_loss += criterion(logits, actions).item()
            all_p.append(torch.argmax(logits,1).cpu())
            all_t.append(actions.cpu())
    all_p = torch.cat(all_p); all_t = torch.cat(all_t)
    acc = (all_p==all_t).float().mean().item()
    m   = compute_metrics(all_p, all_t)
    return total_loss/len(loader), acc, m

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    ds      = ActionAlignedDataset(n_per_action=150)
    n_train = int(0.8*len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    model     = VLNModel(feature_dim=512, num_actions=4).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {'train_loss':[], 'val_loss':[],
               'train_acc':[], 'val_acc':[], 'sr':[], 'spl':[], 'ne':[]}
    best_sr  = 0.0
    NUM_EPOCHS = 25

    print("\n" + "="*55)
    print("  ACTION-ALIGNED TRAINING — MP3D 17DRP5sb8fy")
    print("="*55)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        tl, ta    = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va, m = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        history['sr'].append(m['sr'])
        history['spl'].append(m['spl'])
        history['ne'].append(m['ne'])

        print(f"  Train → Loss:{tl:.4f} Acc:{ta:.3f}")
        print(f"  Val   → Loss:{vl:.4f} Acc:{va:.3f}")
        print(f"  SR={m['sr']:.3f} SPL={m['spl']:.3f} NE={m['ne']:.3f}")
        print(f"  Per-action: ", end="")
        for k,v in m['action_acc'].items():
            print(f"{k}={v:.2f}", end=" ")
        print()

        if m['sr'] >= best_sr:
            best_sr = m['sr']
            torch.save({'model_state': model.state_dict(),
                        'sr': m['sr'], 'spl': m['spl']},
                       "best_model_final.pth")
            print(f"  ✅ Saved! SR={m['sr']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    epochs = range(1, NUM_EPOCHS+1)
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history['sr'],  'g-o', label='SR')
    axes[2].plot(epochs, history['spl'], 'm-o', label='SPL')
    axes[2].set_title('SR & SPL'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.suptitle('CLIP-VLN — Action-Aligned Training', fontsize=13)
    plt.tight_layout()
    plt.savefig("learning_curves_final.png", dpi=150)
    print(f"\nBest SR: {best_sr:.4f}")
    print("Model: best_model_final.pth")
    print("Curves: learning_curves_final.png")
