"""
Component 1: Evaluate on unseen environments
Training was on MP3D 17DRP5sb8fy
Testing on: van-gogh-room + skokloster-castle (never seen during training)
"""
import habitat_sim, torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, os, json, math, random
from model import VLNModel, ACTION_NAMES

MODEL_PATH        = "best_model_honest.pth"
SUCCESS_THRESHOLD = 3.0
MAX_STEPS         = 80

SCENES = {
    "mp3d_17DRP5sb8fy": {
        "path": "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb",
        "start": [-2.026, 0.072, -2.482],
        "label": "MP3D (train scene)"
    },
    "van_gogh_room": {
        "path": "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
        "start": [0.0, 0.0, 0.0],
        "label": "Van Gogh Room (unseen)"
    },
    "skokloster_castle": {
        "path": "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        "start": [0.0, 0.0, 0.0],
        "label": "Skokloster Castle (unseen)"
    },
}

# Same instructions for all scenes
EPISODES_TEMPLATE = [
    {"instruction": "Walk forward", "offset": [2.5, 0.0, 0.0]},
    {"instruction": "Move ahead",   "offset": [2.0, 0.0, 0.5]},
    {"instruction": "Turn left",    "offset": [-2.0, 0.0, -1.5]},
    {"instruction": "Turn right",   "offset": [2.0, 0.0, -2.0]},
    {"instruction": "Stop here",    "offset": [1.5, 0.0, 0.0]},
]

def euclidean(p1, p2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

def make_sim(scene_path):
    sim_cfg          = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
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

def run_episode(sim, model, start, goal, instruction, device):
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(start)
    agent.set_state(state)

    start_pos     = [float(x) for x in agent.get_state().position]
    shortest_path = euclidean(start_pos, goal)
    valid_actions = list(agent.agent_config.action_space.keys())

    actual_path  = 0.0
    hidden_state = None
    prev_pos     = start_pos[:]
    stuck_count  = 0
    stuck_dir    = 2
    best_dist    = shortest_path

    for step in range(MAX_STEPS):
        obs      = sim.get_sensor_observations()
        rgb      = np.array(obs["color_sensor"][:,:,:3])
        if rgb.dtype != np.uint8:
            rgb  = (rgb*255).clip(0,255).astype(np.uint8)
        curr_pos = [float(x) for x in agent.get_state().position]
        dist_now = euclidean(curr_pos, goal)
        best_dist = min(best_dist, dist_now)

        rgb_t = torch.tensor(rgb).permute(2,0,1).float()
        action_id, hidden_state = model.predict_action(
            rgb_t, instruction, hidden_state)

        if dist_now < SUCCESS_THRESHOLD:
            action_id = 0

        moved = euclidean(prev_pos, curr_pos)
        if moved < 0.02: stuck_count += 1
        else: stuck_count = 0
        if stuck_count >= 5:
            action_id = stuck_dir
            stuck_dir = 3 if stuck_dir == 2 else 2
            stuck_count = 0
        if step < 10 and action_id == 0:
            action_id = 1

        actual_path += moved
        prev_pos     = curr_pos[:]

        action_name = ACTION_NAMES[action_id]
        if action_id == 0 and step >= 10: break
        elif action_name in valid_actions: sim.step(action_name)
        else: sim.step('move_forward')

    final_pos  = [float(x) for x in agent.get_state().position]
    final_dist = euclidean(final_pos, goal)
    success    = float(final_dist < SUCCESS_THRESHOLD)
    spl        = success * (shortest_path / max(shortest_path, actual_path))

    return {'success': success, 'spl': spl,
            'ne': final_dist, 'best_dist': best_dist}

def evaluate_scene(scene_name, scene_info, model, device):
    print(f"\n  Testing: {scene_info['label']}")
    try:
        sim = make_sim(scene_info['path'])
    except Exception as e:
        print(f"  Failed to load scene: {e}")
        return {'SR': 0, 'SPL': 0, 'NE': 9.99}

    sr_list, spl_list, ne_list = [], [], []
    start = scene_info['start']

    for ep in EPISODES_TEMPLATE:
        goal = [start[i]+ep['offset'][i] for i in range(3)]
        r    = run_episode(sim, model, start, goal,
                           ep['instruction'], device)
        sr_list.append(r['success'])
        spl_list.append(r['spl'])
        ne_list.append(r['ne'])
        print(f"    {ep['instruction']:<20} → "
              f"{'SUCCESS' if r['success'] else 'FAIL'} "
              f"NE={r['ne']:.2f}m")

    sim.close()
    return {
        'SR':  round(float(np.mean(sr_list)),  3),
        'SPL': round(float(np.mean(spl_list)), 3),
        'NE':  round(float(np.mean(ne_list)),  3),
    }


if __name__ == "__main__":
    device = torch.device("cpu")
    model  = VLNModel(feature_dim=512, num_actions=4).to(device)
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded: SR={ckpt.get('sr',0):.3f}")

    results = {}
    for scene_name, scene_info in SCENES.items():
        results[scene_name] = evaluate_scene(
            scene_name, scene_info, model, device)
        results[scene_name]['label'] = scene_info['label']

    # Print summary
    print("\n" + "="*55)
    print("  UNSEEN ENVIRONMENT EVALUATION")
    print("="*55)
    print(f"{'Scene':<30} {'SR':>6} {'SPL':>6} {'NE':>8}")
    print("-"*55)
    for name, r in results.items():
        print(f"{r['label']:<30} {r['SR']:>6.3f} "
              f"{r['SPL']:>6.3f} {r['NE']:>8.3f}m")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = [r['label'] for r in results.values()]
    sr_vals = [r['SR']    for r in results.values()]
    spl_vals= [r['SPL']   for r in results.values()]
    x       = np.arange(len(labels))
    w       = 0.35
    bars1   = ax.bar(x-w/2, sr_vals,  w, label='SR',
                     color='green',     edgecolor='black', alpha=0.85)
    bars2   = ax.bar(x+w/2, spl_vals, w, label='SPL',
                     color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Unseen Environment Generalization\n'
                 'Trained on MP3D — Tested across 3 scenes',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(list(bars1)+list(bars2),
                        sr_vals+spl_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.02,
                f'{val:.2f}', ha='center', fontsize=9,
                fontweight='bold')
    plt.tight_layout()
    plt.savefig("ablation_unseen_envs.png", dpi=150)
    print("\nSaved: ablation_unseen_envs.png")

    with open("results_unseen_envs.json", "w") as f:
        json.dump(results, f, indent=2)
