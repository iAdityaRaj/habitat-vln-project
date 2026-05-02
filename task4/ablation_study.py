"""
Component 3: Reduced training data ablation
Component 4: Frozen vs fine-tuned CLIP encoders
"""
import habitat_sim, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json, math, random, os, sys
sys.path.insert(0, os.path.abspath(".."))
from task3.model import VLNModel, ACTION_NAMES
from task3.metrics import compute_metrics

SCENE = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

NAVIGABLE = [
    [-2.026, 0.072, -2.482], [-5.084, 0.072, -1.585],
    [-0.785, 0.072,  0.656], [-1.118, 0.072, -1.522],
    [-0.608, 0.072,  1.880], [ 0.085, 0.072,  0.830],
    [-3.748, 0.072, -2.091], [-7.510, 0.072, -0.806],
    [ 0.959, 0.072,  0.450], [-6.150, 0.072,  0.415],
]

TRAIN_INSTRUCTIONS = {
    0: ["Stop here", "Halt now", "Stop at this point"],
    1: ["Walk forward", "Move ahead", "Go straight"],
    2: ["Turn left",  "Go left",   "Rotate left"],
    3: ["Turn right", "Go right",  "Rotate right"],
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
    def __init__(self, n_per_action=80, seed=42):
        random.seed(seed); np.random.seed(seed)
        self.samples = []
        sim   = make_sim()
        agent = sim.initialize_agent(0)
        valid = list(agent.agent_config.action_space.keys())
        for action_id in range(4):
            for _ in range(n_per_action):
                pos   = random.choice(NAVIGABLE)
                state = habitat_sim.AgentState()
                state.position = np.array([
                    pos[0]+random.uniform(-0.2,0.2),
                    pos[1],
                    pos[2]+random.uniform(-0.2,0.2)])
                agent.set_state(state)
                for _ in range(random.randint(0,5)):
                    sim.step(random.choice(valid))
                obs = sim.get_sensor_observations()
                rgb = np.array(obs["color_sensor"][:,:,:3])
                if rgb.dtype != np.uint8:
                    rgb = (rgb*255).clip(0,255).astype(np.uint8)
                instr = random.choice(TRAIN_INSTRUCTIONS[action_id])
                self.samples.append({'rgb':rgb,'instruction':instr,'action':action_id})
        sim.close()
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s['rgb']).permute(2,0,1).float(),
                s['instruction'],
                torch.tensor(s['action'], dtype=torch.long))

def quick_train(model, dataset, epochs=10, lr=1e-4):
    """Quick training for ablation comparison."""
    loader    = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for rgb, instr, actions in loader:
            optimizer.zero_grad()
            logits, _ = model(rgb, list(instr), hidden_state=None)
            loss = criterion(logits, actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    return model

def quick_eval(model, dataset):
    """Quick evaluation."""
    loader    = DataLoader(dataset, batch_size=16, shuffle=False)
    all_p, all_t = [], []
    model.eval()
    with torch.no_grad():
        for rgb, instr, actions in loader:
            logits, _ = model(rgb, list(instr), hidden_state=None)
            all_p.append(torch.argmax(logits,1))
            all_t.append(actions)
    all_p = torch.cat(all_p); all_t = torch.cat(all_t)
    m = compute_metrics(all_p, all_t)
    return m['sr'], m['spl']


# ── ABLATION 1: Reduced Training Data ────────────────────────────────────────
def ablation_data_size():
    print("\n" + "="*55)
    print("  ABLATION 1: REDUCED TRAINING DATA")
    print("="*55)

    print("Collecting full dataset...")
    full_dataset = VLNDataset(n_per_action=80)
    n_full       = len(full_dataset)

    data_fractions = [1.0, 0.75, 0.50, 0.25]
    results        = []

    for frac in data_fractions:
        n_samples = max(16, int(n_full * frac))
        # Subset
        indices = list(range(n_samples))
        subset  = torch.utils.data.Subset(full_dataset, indices)

        model = VLNModel(feature_dim=512, num_actions=4)
        model = quick_train(model, subset, epochs=10)

        # Eval on held-out
        val_indices = list(range(n_full-80, n_full))
        val_set     = torch.utils.data.Subset(full_dataset, val_indices)
        sr, spl     = quick_eval(model, val_set)

        results.append({
            'fraction':  frac,
            'n_samples': n_samples,
            'SR':        round(sr,  3),
            'SPL':       round(spl, 3),
        })
        print(f"  {frac*100:.0f}% data ({n_samples:4d} samples) "
              f"→ SR={sr:.3f} SPL={spl:.3f}")

    return results


# ── ABLATION 2: Frozen vs Fine-tuned CLIP ────────────────────────────────────
def ablation_frozen_vs_finetuned():
    print("\n" + "="*55)
    print("  ABLATION 2: FROZEN vs FINE-TUNED CLIP")
    print("="*55)

    print("Collecting dataset...")
    dataset = VLNDataset(n_per_action=80)
    n       = len(dataset)
    train_set = torch.utils.data.Subset(dataset, list(range(n-80)))
    val_set   = torch.utils.data.Subset(dataset, list(range(n-80, n)))

    results = {}

    # Condition A: Frozen CLIP (default — what we trained)
    print("\n  Condition A: FROZEN CLIP encoders")
    model_frozen = VLNModel(feature_dim=512, num_actions=4)
    # Verify encoders are frozen
    trainable = sum(p.numel() for p in model_frozen.parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in model_frozen.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)")
    model_frozen = quick_train(model_frozen, train_set, epochs=15)
    sr_f, spl_f  = quick_eval(model_frozen, val_set)
    results['frozen'] = {'SR': round(sr_f,3), 'SPL': round(spl_f,3),
                          'trainable_params': trainable}
    print(f"  Frozen  → SR={sr_f:.3f} SPL={spl_f:.3f}")

    # Condition B: Fine-tuned CLIP (unfreeze encoders)
    print("\n  Condition B: FINE-TUNED CLIP encoders")
    model_ft = VLNModel(feature_dim=512, num_actions=4)
    # Unfreeze CLIP
    for param in model_ft.visual_encoder.clip_model.parameters():
        param.requires_grad = True
    for param in model_ft.text_encoder.clip_model.parameters():
        param.requires_grad = True
    trainable_ft = sum(p.numel() for p in model_ft.parameters()
                       if p.requires_grad)
    print(f"  Trainable params: {trainable_ft:,} / {total:,} "
          f"({trainable_ft/total*100:.1f}%)")
    # Use smaller LR for fine-tuning
    model_ft = quick_train(model_ft, train_set, epochs=15, lr=1e-5)
    sr_ft, spl_ft = quick_eval(model_ft, val_set)
    results['finetuned'] = {'SR': round(sr_ft,3), 'SPL': round(spl_ft,3),
                             'trainable_params': trainable_ft}
    print(f"  Fine-tuned → SR={sr_ft:.3f} SPL={spl_ft:.3f}")

    return results


# ── Plot all ablations ────────────────────────────────────────────────────────
def plot_ablations(data_results, frozen_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Data size ablation
    fracs  = [f"{r['fraction']*100:.0f}%" for r in data_results]
    sr_vals= [r['SR']  for r in data_results]
    spl_vals=[r['SPL'] for r in data_results]
    axes[0].plot(fracs, sr_vals,  'g-o', linewidth=2,
                 markersize=8, label='SR')
    axes[0].plot(fracs, spl_vals, 'm-o', linewidth=2,
                 markersize=8, label='SPL')
    axes[0].set_title('Ablation: Training Data Size',
                      fontweight='bold')
    axes[0].set_xlabel('% of Training Data')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for i, (s, sp) in enumerate(zip(sr_vals, spl_vals)):
        axes[0].annotate(f'{s:.2f}', (fracs[i], s),
                         textcoords="offset points",
                         xytext=(0,8), ha='center', fontsize=9)

    # Plot 2: Frozen vs finetuned
    conditions = ['Frozen\nCLIP', 'Fine-tuned\nCLIP']
    sr2  = [frozen_results['frozen']['SR'],
            frozen_results['finetuned']['SR']]
    spl2 = [frozen_results['frozen']['SPL'],
            frozen_results['finetuned']['SPL']]
    x = np.arange(len(conditions))
    w = 0.35
    b1 = axes[1].bar(x-w/2, sr2,  w, label='SR',
                     color=['steelblue','orange'],
                     edgecolor='black', alpha=0.85)
    b2 = axes[1].bar(x+w/2, spl2, w, label='SPL',
                     color=['steelblue','orange'],
                     edgecolor='black', alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Ablation: Frozen vs Fine-tuned CLIP',
                      fontweight='bold')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(list(b1)+list(b2), sr2+spl2):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.02,
                     f'{val:.2f}', ha='center',
                     fontsize=10, fontweight='bold')

    # Plot 3: Summary comparison
    exp_names = ['100%\nData\nFrozen', '50%\nData\nFrozen',
                 '25%\nData\nFrozen', '100%\nData\nFinetuned']
    exp_sr = [data_results[0]['SR'], data_results[2]['SR'],
              data_results[3]['SR'], frozen_results['finetuned']['SR']]
    colors = ['green','steelblue','tomato','orange']
    bars   = axes[2].bar(exp_names, exp_sr, color=colors,
                         edgecolor='black', alpha=0.85)
    axes[2].set_title('Ablation Summary — SR Comparison',
                      fontweight='bold')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, exp_sr):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.02,
                     f'{val:.2f}', ha='center',
                     fontsize=10, fontweight='bold')

    plt.suptitle('CLIP-VLN Ablation Studies',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("ablation_study.png", dpi=150, bbox_inches='tight')
    print("\nSaved: ablation_study.png")
    plt.close()


if __name__ == "__main__":
    data_results   = ablation_data_size()
    frozen_results = ablation_frozen_vs_finetuned()
    plot_ablations(data_results, frozen_results)

    with open("results_ablation.json", "w") as f:
        json.dump({
            'data_size_ablation': data_results,
            'frozen_vs_finetuned': frozen_results,
        }, f, indent=2)
    print("Saved: results_ablation.json")
    print("open ablation_study.png")
