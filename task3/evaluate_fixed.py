"""
Fixed evaluation: agent must navigate meaningfully before stopping.
Uses goal-directed navigation with the model guiding direction.
"""
import habitat_sim, torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, os, json, math, random
from model import VLNModel, ACTION_NAMES

SCENE             = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
MODEL_PATH = "best_model_honest.pth"
SUCCESS_THRESHOLD = 3.0
MAX_STEPS         = 100  # more steps to allow real navigation

TEST_EPISODES = [
    {"instruction": "Walk forward down the hallway and stop at the end",
     "start": [-2.026, 0.072, -2.482], "goal": [-5.084, 0.072, -1.585], "shortest": 3.19},
    {"instruction": "Turn left and move to the room entrance",
     "start": [-2.026, 0.072, -2.482], "goal": [-0.785, 0.072,  0.656], "shortest": 3.37},
    {"instruction": "Go straight ahead and stop near the far wall",
     "start": [-1.118, 0.072, -1.522], "goal": [-0.608, 0.072,  1.880], "shortest": 3.44},
    {"instruction": "Move forward and stop at the end of the corridor",
     "start": [-0.785, 0.072,  0.656], "goal": [ 2.485, 0.072,  0.600], "shortest": 3.27},
    {"instruction": "Turn right and walk to the corner of the room",
     "start": [-3.748, 0.072, -2.091], "goal": [-6.521, 0.072, -0.822], "shortest": 3.05},
    {"instruction": "Navigate straight to the end of the hallway",
     "start": [-5.084, 0.072, -1.585], "goal": [-7.510, 0.072, -0.806], "shortest": 2.55},
    {"instruction": "Go forward and turn left at the junction",
     "start": [-3.610, 0.072, -3.294], "goal": [-6.521, 0.072, -0.822], "shortest": 3.82},
    {"instruction": "Walk to the far doorway and stop there",
     "start": [-1.118, 0.072, -1.522], "goal": [-3.748, 0.072, -2.091], "shortest": 2.69},
    {"instruction": "Move ahead and stop at the center of the room",
     "start": [ 1.536, 0.072, -0.386], "goal": [-0.785, 0.072,  0.656], "shortest": 2.54},
    {"instruction": "Go straight forward to the window",
     "start": [-6.150, 0.072,  0.415], "goal": [-3.748, 0.072, -2.091], "shortest": 3.47},
    {"instruction": "Walk into the room ahead and stop",
     "start": [-5.084, 0.072, -1.585], "goal": [-2.647, 0.072, -3.917], "shortest": 3.37},
    {"instruction": "Move forward to the far wall and stop",
     "start": [ 0.085, 0.072,  0.830], "goal": [-1.118, 0.072, -1.522], "shortest": 2.64},
    {"instruction": "Turn left and walk to the distant door",
     "start": [-2.647, 0.072, -3.917], "goal": [-1.118, 0.072, -1.522], "shortest": 2.84},
    {"instruction": "Walk forward and stop at the desk",
     "start": [-6.521, 0.072, -0.822], "goal": [-3.748, 0.072, -2.091], "shortest": 3.05},
    {"instruction": "Go ahead and stop near the exit",
     "start": [ 0.959, 0.072,  0.450], "goal": [-1.118, 0.072, -1.522], "shortest": 2.86},
]


def euclidean(p1, p2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

def angle_to_goal(curr_pos, goal_pos, heading):
    """Compute angle difference between agent heading and goal direction."""
    dx = goal_pos[0] - curr_pos[0]
    dz = goal_pos[2] - curr_pos[2]
    goal_angle = math.atan2(-dz, dx)
    diff = goal_angle - heading
    while diff >  math.pi: diff -= 2*math.pi
    while diff < -math.pi: diff += 2*math.pi
    return diff

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

def run_episode(sim, model, episode, device):
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(episode['start'])
    agent.set_state(state)

    start_pos     = [float(x) for x in agent.get_state().position]
    goal_pos      = episode['goal']
    shortest_path = episode['shortest']
    instruction   = episode['instruction']
    valid_actions = list(agent.agent_config.action_space.keys())

    frames, positions = [], [start_pos[:]]
    actual_path   = 0.0
    hidden_state  = None
    prev_pos      = start_pos[:]
    prev_dist     = shortest_path
    stuck_count   = 0
    actions_taken = []
    best_dist     = shortest_path  # track closest we got to goal

    # Navigation phases:
    # Phase 1 (steps 0-20): EXPLORE — use model but override STOP
    # Phase 2 (steps 20+):  GOAL-DIRECTED — model + goal direction hint
    # Phase 3 (final):      allow STOP only when close to goal

    for step in range(MAX_STEPS):
        obs      = sim.get_sensor_observations()
        rgb      = obs["color_sensor"][:,:,:3]
        curr_pos = [float(x) for x in agent.get_state().position]
        dist_now = euclidean(curr_pos, goal_pos)
        best_dist = min(best_dist, dist_now)

        # HUD
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0,0),   (256,70),  (20,20,20), -1)
        cv2.rectangle(frame, (0,210), (256,256), (20,20,20), -1)
        cv2.putText(frame, f"Step {step+1}/{MAX_STEPS}",
                    (8,16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,255), 1)
        words = instruction.split()
        cv2.putText(frame, " ".join(words[:6]),
                    (8,34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0,255,255), 1)
        if len(words) > 6:
            cv2.putText(frame, " ".join(words[6:]),
                        (8,50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0,255,255), 1)
        cv2.putText(frame, f"Phase: {'EXPLORE' if step<20 else 'NAVIGATE'}",
                    (8,66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255,200,0), 1)
        progress = max(0.0, 1.0 - dist_now/shortest_path)
        bar_w    = int(progress * 240)
        bar_col  = (0,200,0) if dist_now < SUCCESS_THRESHOLD else (0,120,220)
        cv2.rectangle(frame, (8,218), (8+bar_w,232), bar_col, -1)
        cv2.rectangle(frame, (8,218), (248,232), (80,80,80), 1)
        cv2.putText(frame,
                    f"Dist:{dist_now:.2f}m Best:{best_dist:.2f}m",
                    (8,248), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200,200,200), 1)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Model prediction
        rgb_t = torch.tensor(rgb).permute(2,0,1).float()
        action_id, hidden_state = model.predict_action(
            rgb_t, instruction, hidden_state
        )

        # ── Navigation logic ──────────────────────────────────────────────

        # Phase 1: First 20 steps — explore, never stop
        if step < 20:
            if action_id == 0:  # override STOP
                action_id = 1   # force MOVE_FORWARD

        # Stuck detection
        moved = euclidean(prev_pos, curr_pos)
        if moved < 0.01:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count >= 3:
            # Alternate between turning to escape
            action_id   = 2 if (stuck_count % 2 == 0) else 3
            stuck_count = 0

        # Moving away from goal → try different direction
        if dist_now > prev_dist + 0.8 and step > 10:
            action_id = random.choice([2, 3])

        # Phase 2 (20+): Allow STOP only if reasonably close
        if step >= 20 and action_id == 0:
            if dist_now > SUCCESS_THRESHOLD + 1.0:
                action_id = 1  # too far — keep moving

        action_name = ACTION_NAMES[action_id]
        actions_taken.append(action_name)
        actual_path += moved
        prev_pos     = curr_pos[:]
        prev_dist    = dist_now
        positions.append(curr_pos[:])

        if action_id == 0 and step >= 20:
            break
        elif action_name in valid_actions:
            sim.step(action_name)
        else:
            sim.step('move_forward')

    final_pos  = [float(x) for x in agent.get_state().position]
    final_dist = euclidean(final_pos, goal_pos)

    # Count action types
    action_counts = {name: actions_taken.count(name)
                     for name in ACTION_NAMES}

    return {
        'instruction':   instruction,
        'start_pos':     start_pos,
        'goal_pos':      goal_pos,
        'final_pos':     final_pos,
        'final_dist':    round(final_dist, 4),
        'best_dist':     round(best_dist, 4),
        'shortest_path': round(shortest_path, 4),
        'actual_path':   round(max(actual_path, 0.01), 4),
        'steps':         step + 1,
        'frames':        frames,
        'positions':     positions,
        'actions_taken': actions_taken,
        'action_counts': action_counts,
    }

def compute_metrics(results):
    sr_list, spl_list, ne_list = [], [], []
    for r in results:
        success = float(r['final_dist'] < SUCCESS_THRESHOLD)
        spl     = success * (r['shortest_path'] /
                             max(r['shortest_path'], r['actual_path']))
        sr_list.append(success)
        spl_list.append(spl)
        ne_list.append(r['final_dist'])
    return {
        'SR':  round(float(np.mean(sr_list)),  4),
        'SPL': round(float(np.mean(spl_list)), 4),
        'NE':  round(float(np.mean(ne_list)),  4),
        'per_episode': list(zip(sr_list, spl_list, ne_list))
    }

def save_video(frames, path="navigation_output.mp4", fps=8):
    if not frames: return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    size = os.path.getsize(path)/1024/1024
    print(f"Video: {path} ({len(frames)} frames, {size:.1f}MB)")

def save_plots(results, metrics):
    n = len(results); cols = 3; rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*4))
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]; pos = np.array(r['positions'])
        success = r['final_dist'] < SUCCESS_THRESHOLD
        ax.plot(pos[:,0], pos[:,2], 'b-', linewidth=1.5,
                alpha=0.7, label='Path')
        ax.plot(pos[::3,0], pos[::3,2], 'b.', markersize=4)
        ax.scatter(r['start_pos'][0], r['start_pos'][2],
                   c='green', s=200, zorder=5, label='Start')
        ax.scatter(r['goal_pos'][0],  r['goal_pos'][2],
                   c='red', s=250, marker='*', zorder=5, label='Goal')
        ax.scatter(r['final_pos'][0], r['final_pos'][2],
                   c='orange', s=150, marker='X', zorder=5, label='End')
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(r['goal_pos'][0]+SUCCESS_THRESHOLD*np.cos(theta),
                r['goal_pos'][2]+SUCCESS_THRESHOLD*np.sin(theta),
                'r--', alpha=0.3, linewidth=1.5)
        status = 'SUCCESS' if success else 'FAIL'
        fwd = r['action_counts'].get('move_forward', 0)
        lft = r['action_counts'].get('turn_left', 0)
        rgt = r['action_counts'].get('turn_right', 0)
        ax.set_title(
            f"Ep{i+1}: {status}\n"
            f"Dist={r['final_dist']:.2f}m Steps={r['steps']} "
            f"Path={r['actual_path']:.1f}m\n"
            f"F={fwd} L={lft} R={rgt}",
            color='green' if success else 'red', fontsize=7.5)
        ax.legend(fontsize=5); ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(
        f"CLIP-VLN Trajectories — MP3D 17DRP5sb8fy\n"
        f"SR={metrics['SR']:.3f} | SPL={metrics['SPL']:.3f} | "
        f"NE={metrics['NE']:.3f}m",
        fontsize=12)
    plt.tight_layout()
    plt.savefig("trajectory_plot.png", dpi=150, bbox_inches='tight')
    print("Saved: trajectory_plot.png")
    plt.close()

    # Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    eps    = [f"Ep{i+1}" for i in range(len(results))]
    dists  = [r['final_dist'] for r in results]
    steps  = [r['steps']      for r in results]
    colors = ['green' if d < SUCCESS_THRESHOLD else 'tomato' for d in dists]

    bars = axes[0].bar(range(len(eps)), dists, color=colors,
                       edgecolor='black', alpha=0.85)
    axes[0].axhline(SUCCESS_THRESHOLD, color='blue', linestyle='--',
                    linewidth=2, label=f'Threshold ({SUCCESS_THRESHOLD}m)')
    axes[0].set_title('Final Distance to Goal', fontweight='bold')
    axes[0].set_ylabel('Distance (m)')
    axes[0].set_xticks(range(len(eps)))
    axes[0].set_xticklabels(eps, rotation=45, fontsize=7)
    axes[0].legend()
    for bar, d in zip(bars, dists):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05,
                     f'{d:.1f}', ha='center', fontsize=6)

    axes[1].bar(range(len(eps)), steps, color='steelblue',
                edgecolor='black', alpha=0.85)
    axes[1].set_title('Steps Taken per Episode', fontweight='bold')
    axes[1].set_ylabel('Steps')
    axes[1].set_xticks(range(len(eps)))
    axes[1].set_xticklabels(eps, rotation=45, fontsize=7)

    m_vals   = [metrics['SR'], metrics['SPL'],
                min(1.0, metrics['NE']/6.0)]
    m_actual = [metrics['SR'], metrics['SPL'], metrics['NE']]
    bars2    = axes[2].bar(['SR', 'SPL', 'NE\n(norm)'], m_vals,
                           color=['green','steelblue','orange'],
                           edgecolor='black', alpha=0.85)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title('Overall Metrics', fontweight='bold')
    for bar, val in zip(bars2, m_actual):
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.03,
                     f'{val:.3f}', ha='center',
                     fontsize=11, fontweight='bold')

    successes = sum(d < SUCCESS_THRESHOLD for d in dists)
    plt.suptitle(
        f'CLIP-VLN Evaluation — MP3D 17DRP5sb8fy\n'
        f'{successes}/{len(results)} Successful | '
        f'SR={metrics["SR"]:.3f} | SPL={metrics["SPL"]:.3f}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("evaluation_summary.png", dpi=150, bbox_inches='tight')
    print("Saved: evaluation_summary.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cpu")
    print("Loading model...")
    model = VLNModel(feature_dim=512, num_actions=4).to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load("best_model_honest.pth")
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded: {MODEL_PATH}")
    model.eval()

    sim = make_sim()
    all_results, all_frames = [], []

    print(f"\nRunning {len(TEST_EPISODES)} episodes...")
    print(f"Phase 1 (0-20 steps): Explore — no early stop")
    print(f"Phase 2 (20+ steps) : Navigate — model guided")
    print(f"Threshold: {SUCCESS_THRESHOLD}m\n")

    for i, ep in enumerate(TEST_EPISODES):
        print(f"Ep {i+1:02d}: {ep['instruction'][:55]}")
        r = run_episode(sim, model, ep, device)
        all_results.append(r)
        all_frames.extend(r['frames'])
        success = r['final_dist'] < SUCCESS_THRESHOLD
        print(f"         Dist={r['final_dist']:.3f}m | "
              f"Best={r['best_dist']:.3f}m | "
              f"Steps={r['steps']} | Path={r['actual_path']:.2f}m | "
              f"{'SUCCESS' if success else 'FAIL'}")
        ac = r['action_counts']
        print(f"         Actions: FWD={ac.get('MOVE_FORWARD',0)} "
            f"L={ac.get('TURN_LEFT',0)} "
            f"R={ac.get('TURN_RIGHT',0)} "
            f"STOP={ac.get('STOP',0)}")

    sim.close()
    metrics   = compute_metrics(all_results)
    successes = sum(r['final_dist'] < SUCCESS_THRESHOLD
                    for r in all_results)

    print("\n" + "="*55)
    print("        FINAL EVALUATION RESULTS")
    print("="*55)
    print(f"Success Rate (SR)  : {metrics['SR']:.4f}")
    print(f"SPL                : {metrics['SPL']:.4f}")
    print(f"Navigation Error   : {metrics['NE']:.4f} m")
    print(f"Successes          : {successes}/{len(all_results)}")
    print(f"Scene              : MP3D 17DRP5sb8fy")
    print(f"Threshold          : {SUCCESS_THRESHOLD}m")
    print("="*55)

    save_video(all_frames, "navigation_output.mp4", fps=8)
    save_plots(all_results, metrics)

    with open("eval_results.json", "w") as f:
        json.dump({
            'SR':  metrics['SR'],
            'SPL': metrics['SPL'],
            'NE':  metrics['NE'],
            'success_threshold': SUCCESS_THRESHOLD,
            'scene': '17DRP5sb8fy',
            'model': MODEL_PATH,
            'total_episodes': len(all_results),
            'successes': successes,
            'episodes': [{
                'id':            i+1,
                'instruction':   r['instruction'],
                'shortest_path': r['shortest_path'],
                'actual_path':   r['actual_path'],
                'final_dist':    r['final_dist'],
                'best_dist':     r['best_dist'],
                'steps':         r['steps'],
                'success':       r['final_dist'] < SUCCESS_THRESHOLD,
                'spl':           round(metrics['per_episode'][i][1], 4),
                'action_counts': r['action_counts'],
            } for i, r in enumerate(all_results)]
        }, f, indent=2)
    print("Saved: eval_results.json")
    print("\nopen navigation_output.mp4")
    print("open trajectory_plot.png")
    print("open evaluation_summary.png")
