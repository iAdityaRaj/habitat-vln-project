"""
Component 2: Paraphrased instruction evaluation
Same navigation task, different wording
"""
import habitat_sim, torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, math, os
from model import VLNModel, ACTION_NAMES

MODEL_PATH        = "best_model_honest.pth"
SCENE             = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
SUCCESS_THRESHOLD = 3.0
MAX_STEPS         = 80

# Same 5 episodes, 3 instruction versions each
PARAPHRASE_EPISODES = [
    {
        "start": [-5.084, 0.072, -1.585],
        "goal":  [-2.026, 0.072, -2.482],
        "shortest": 3.19,
        "versions": {
            "original":    "Walk forward",
            "paraphrase1": "Move ahead",
            "paraphrase2": "Head in the forward direction",
        }
    },
    {
        "start": [-2.026, 0.072, -2.482],
        "goal":  [-5.084, 0.072, -1.585],
        "shortest": 3.19,
        "versions": {
            "original":    "Turn left",
            "paraphrase1": "Make a left",
            "paraphrase2": "Veer to the left side",
        }
    },
    {
        "start": [-5.084, 0.072, -1.585],
        "goal":  [-2.026, 0.072, -2.482],
        "shortest": 3.19,
        "versions": {
            "original":    "Turn right",
            "paraphrase1": "Make a right",
            "paraphrase2": "Veer to the right side",
        }
    },
    {
        "start": [-1.118, 0.072, -1.522],
        "goal":  [-0.785, 0.072,  0.656],
        "shortest": 3.44,
        "versions": {
            "original":    "Stop here",
            "paraphrase1": "You have arrived at your destination",
            "paraphrase2": "Halt now",
        }
    },
    {
        "start": [-6.521, 0.072, -0.822],
        "goal":  [-3.748, 0.072, -2.091],
        "shortest": 3.05,
        "versions": {
            "original":    "Walk forward and stop",
            "paraphrase1": "Go straight forward and stop",
            "paraphrase2": "Proceed ahead then halt",
        }
    },
]

def euclidean(p1, p2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

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

def run_episode(sim, model, ep, instruction, device):
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(ep['start'])
    agent.set_state(state)

    start_pos     = [float(x) for x in agent.get_state().position]
    goal_pos      = ep['goal']
    shortest_path = ep['shortest']
    valid_actions = list(agent.agent_config.action_space.keys())

    actual_path  = 0.0
    hidden_state = None
    prev_pos     = start_pos[:]
    stuck_count  = 0
    stuck_dir    = 2

    for step in range(MAX_STEPS):
        obs      = sim.get_sensor_observations()
        rgb      = np.array(obs["color_sensor"][:,:,:3])
        if rgb.dtype != np.uint8:
            rgb  = (rgb*255).clip(0,255).astype(np.uint8)
        curr_pos = [float(x) for x in agent.get_state().position]
        dist_now = euclidean(curr_pos, goal_pos)

        rgb_t = torch.tensor(rgb).permute(2,0,1).float()
        action_id, hidden_state = model.predict_action(
            rgb_t, instruction, hidden_state)

        if dist_now < SUCCESS_THRESHOLD: action_id = 0
        moved = euclidean(prev_pos, curr_pos)
        if moved < 0.02: stuck_count += 1
        else: stuck_count = 0
        if stuck_count >= 5:
            action_id = stuck_dir
            stuck_dir = 3 if stuck_dir == 2 else 2
            stuck_count = 0
        if step < 10 and action_id == 0: action_id = 1

        actual_path += moved
        prev_pos     = curr_pos[:]

        action_name = ACTION_NAMES[action_id]
        if action_id == 0 and step >= 10: break
        elif action_name in valid_actions: sim.step(action_name)
        else: sim.step('move_forward')

    final_pos  = [float(x) for x in agent.get_state().position]
    final_dist = euclidean(final_pos, goal_pos)
    success    = float(final_dist < SUCCESS_THRESHOLD)
    spl        = success * (shortest_path / max(shortest_path, actual_path))
    return {'success': success, 'spl': spl, 'ne': final_dist}


if __name__ == "__main__":
    device = torch.device("cpu")
    model  = VLNModel(feature_dim=512, num_actions=4).to(device)
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    sim = make_sim()
    results = {'original': [], 'paraphrase1': [], 'paraphrase2': []}

    print("="*60)
    print("  PARAPHRASE EVALUATION")
    print("="*60)

    for i, ep in enumerate(PARAPHRASE_EPISODES):
        print(f"\nEpisode {i+1}:")
        for version, instruction in ep['versions'].items():
            r = run_episode(sim, model, ep, instruction, device)
            results[version].append(r)
            print(f"  [{version:12s}] '{instruction[:35]}'"
                  f" → {'SUCCESS' if r['success'] else 'FAIL'}"
                  f" NE={r['ne']:.2f}m")

    sim.close()

    # Compute per-version metrics
    summary = {}
    for version, res in results.items():
        summary[version] = {
            'SR':  round(float(np.mean([r['success'] for r in res])), 3),
            'SPL': round(float(np.mean([r['spl']     for r in res])), 3),
            'NE':  round(float(np.mean([r['ne']      for r in res])), 3),
        }

    print("\n" + "="*55)
    print("  PARAPHRASE RESULTS SUMMARY")
    print("="*55)
    print(f"{'Version':<15} {'SR':>6} {'SPL':>6} {'NE':>8}")
    print("-"*40)
    for version, m in summary.items():
        print(f"{version:<15} {m['SR']:>6.3f} "
              f"{m['SPL']:>6.3f} {m['NE']:>8.3f}m")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    versions  = list(summary.keys())
    sr_vals   = [summary[v]['SR']  for v in versions]
    spl_vals  = [summary[v]['SPL'] for v in versions]
    ne_vals   = [summary[v]['NE']  for v in versions]
    colors    = ['steelblue', 'green', 'orange']

    for ax, vals, title, ylabel in zip(
            axes,
            [sr_vals, spl_vals, ne_vals],
            ['Success Rate (SR)', 'SPL', 'Navigation Error (NE)'],
            ['SR', 'SPL', 'NE (m)']):
        bars = ax.bar(versions, vals, color=colors,
                      edgecolor='black', alpha=0.85)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        if title != 'Navigation Error (NE)':
            ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.02,
                    f'{val:.2f}', ha='center',
                    fontsize=10, fontweight='bold')

    plt.suptitle('Paraphrase Robustness — Same Task, Different Wording',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig("ablation_paraphrase.png", dpi=150)
    print("\nSaved: ablation_paraphrase.png")

    with open("results_paraphrase.json", "w") as f:
        json.dump(summary, f, indent=2)
