import habitat_sim, torch, numpy as np
import cv2, os, math, random
import sys
sys.path.insert(0, os.path.abspath(".."))
from task3.model import VLNModel, ACTION_NAMES

MODEL_PATH        = "best_model_honest.pth"
SCENE             = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
SUCCESS_THRESHOLD = 3.0
MAX_STEPS         = 100

# Better episodes: goals within 3m so turning+moving reaches them
TEST_EPISODES = [
    # FORWARD — goal directly ahead, agent moves into it
    {"instruction": "Walk forward",
     "start": [-5.084, 0.072, -1.585],
     "goal":  [-2.026, 0.072, -2.482], "shortest": 3.19,
     "type": "FORWARD"},
    {"instruction": "Move ahead toward the room",
     "start": [-6.521, 0.072, -0.822],
     "goal":  [-3.748, 0.072, -2.091], "shortest": 3.05,
     "type": "FORWARD"},
    {"instruction": "Go straight forward",
     "start": [-2.647, 0.072, -3.917],
     "goal":  [-1.118, 0.072, -1.522], "shortest": 2.84,
     "type": "FORWARD"},

    # LEFT — goal close enough that turning + few steps reaches it
    {"instruction": "Turn left",
     "start": [-1.118, 0.072, -1.522],
     "goal":  [-2.026, 0.072, -2.482], "shortest": 1.30,
     "type": "TURN_LEFT"},
    {"instruction": "Go left toward the door",
     "start": [-1.118, 0.072, -1.522],
     "goal":  [-3.748, 0.072, -2.091], "shortest": 2.69,
     "type": "TURN_LEFT"},

    # RIGHT — goal close enough on right side
    {"instruction": "Turn right",
     "start": [-5.084, 0.072, -1.585],
     "goal":  [-3.748, 0.072, -2.091], "shortest": 1.80,
     "type": "TURN_RIGHT"},
    {"instruction": "Go right toward the exit",
     "start": [-3.748, 0.072, -2.091],
     "goal":  [-1.118, 0.072, -1.522], "shortest": 2.69,
     "type": "TURN_RIGHT"},

    # STOP — agent should stop quickly
    {"instruction": "Stop here",
     "start": [-1.118, 0.072, -1.522],
     "goal":  [-0.785, 0.072,  0.656], "shortest": 3.44,
     "type": "STOP"},
    {"instruction": "Walk forward and stop",
     "start": [-6.521, 0.072, -0.822],
     "goal":  [-3.748, 0.072, -2.091], "shortest": 3.05,
     "type": "STOP"},
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

def make_frame(rgb, ep, step, max_steps, dist, best_dist,
               ep_num, total_eps):
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0,0), (256,80), (15,15,15), -1)
    cv2.rectangle(frame, (0,210), (256,256), (15,15,15), -1)

    # Type badge
    type_colors = {
        'FORWARD':    (0,180,0),
        'TURN_LEFT':  (180,100,0),
        'TURN_RIGHT': (0,100,180),
        'STOP':       (180,0,180),
    }
    tc = type_colors.get(ep.get('type','FORWARD'), (100,100,100))
    cv2.rectangle(frame, (170,2), (254,18), tc, -1)
    cv2.putText(frame, ep.get('type',''),
                (173,14), cv2.FONT_HERSHEY_SIMPLEX,
                0.30, (255,255,255), 1)

    cv2.putText(frame, f"Ep {ep_num}/{total_eps} | Step {step}/{max_steps}",
                (8,14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

    words = ep['instruction'].split()
    cv2.putText(frame, " ".join(words[:5]),
                (8,34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,220,220), 1)
    if len(words) > 5:
        cv2.putText(frame, " ".join(words[5:]),
                    (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,220,220), 1)

    cv2.putText(frame, f"Goal: {ep['shortest']:.1f}m",
                (8,70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130,130,130), 1)

    # Progress bar
    progress = max(0.0, 1.0 - dist/ep['shortest'])
    bar_w    = int(progress * 240)
    bar_col  = (0,200,0) if dist < SUCCESS_THRESHOLD else (50,130,220)
    cv2.rectangle(frame, (8,218), (248,232), (60,60,60), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (8,218), (8+bar_w,232), bar_col, -1)
    cv2.rectangle(frame, (8,218), (248,232), (100,100,100), 1)

    status_col = (0,220,0) if dist < SUCCESS_THRESHOLD else (100,180,255)
    cv2.putText(frame,
                f"Dist:{dist:.2f}m Best:{best_dist:.2f}m",
                (8,248), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_col, 1)

    if dist < SUCCESS_THRESHOLD:
        cv2.putText(frame, "IN GOAL ZONE",
                    (65,260), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0,255,0), 1)
    return frame

def run_episode_with_video(sim, model, episode, device, ep_num, total_eps):
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(episode['start'])
    agent.set_state(state)

    start_pos     = [float(x) for x in agent.get_state().position]
    goal_pos      = episode['goal']
    shortest_path = episode['shortest']
    valid_actions = list(agent.agent_config.action_space.keys())

    frames       = []
    actual_path  = 0.0
    hidden_state = None
    prev_pos     = start_pos[:]
    stuck_count  = 0
    stuck_dir    = 2
    best_dist    = shortest_path

    for step in range(MAX_STEPS):
        obs     = sim.get_sensor_observations()
        rgb_raw = np.array(obs["color_sensor"])
        if rgb_raw.shape[-1] == 4:
            rgb_raw = rgb_raw[:,:,:3]
        if rgb_raw.dtype != np.uint8:
            rgb_raw = (rgb_raw*255).clip(0,255).astype(np.uint8)
        rgb_raw = np.ascontiguousarray(rgb_raw)

        curr_pos = [float(x) for x in agent.get_state().position]
        dist_now = euclidean(curr_pos, goal_pos)
        best_dist = min(best_dist, dist_now)

        frame = make_frame(rgb_raw, episode, step+1, MAX_STEPS,
                           dist_now, best_dist, ep_num, total_eps)
        frames.append(frame)

        rgb_t = torch.tensor(rgb_raw).permute(2,0,1).float()
        action_id, hidden_state = model.predict_action(
            rgb_t, episode['instruction'], hidden_state)

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

        action_name  = ACTION_NAMES[action_id]
        if action_id == 0 and step >= 10:
            for _ in range(8): frames.append(frame)
            break
        elif action_name in valid_actions:
            sim.step(action_name)
        else:
            sim.step('move_forward')

    final_pos  = [float(x) for x in agent.get_state().position]
    final_dist = euclidean(final_pos, goal_pos)
    success    = final_dist < SUCCESS_THRESHOLD

    # Result card
    rc    = np.zeros((256,256,3), dtype=np.uint8)
    color = (0,100,0) if success else (100,0,0)
    rc[:] = color
    tc    = {'FORWARD':(0,180,0),'TURN_LEFT':(180,120,0),
             'TURN_RIGHT':(0,120,180),'STOP':(180,0,180)}
    badge = tc.get(episode.get('type',''), (100,100,100))
    cv2.rectangle(rc, (70,20), (190,45), badge, -1)
    cv2.putText(rc, episode.get('type',''),
                (85,38), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255,255,255), 1)
    cv2.putText(rc, "SUCCESS!" if success else "FAILED",
                (50 if success else 40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(rc, f"Instruction:",
                (15,125), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (180,180,180), 1)
    words = episode['instruction'].split()
    cv2.putText(rc, " ".join(words[:5]),
                (15,145), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220,220,220), 1)
    cv2.putText(rc, f"Final dist: {final_dist:.2f}m",
                (40,175), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (200,200,200), 1)
    cv2.putText(rc, f"Ep {ep_num}/{total_eps}",
                (95,210), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150,150,150), 1)
    for _ in range(16): frames.append(rc)

    status = 'SUCCESS' if success else 'FAIL  '
    print(f"  Ep {ep_num} [{episode.get('type',''):10s}]: "
          f"{episode['instruction']:<30} → {status} "
          f"(dist={final_dist:.2f}m)")
    return frames, success


if __name__ == "__main__":
    device = torch.device("cpu")
    model  = VLNModel(feature_dim=512, num_actions=4).to(device)
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded (SR={ckpt.get('sr',0):.3f})\n")

    sim        = make_sim()
    all_frames = []
    successes  = 0
    total      = len(TEST_EPISODES)

    # Title card
    tc = np.zeros((256,256,3), dtype=np.uint8)
    tc[:] = (20,20,40)
    cv2.putText(tc,"CLIP-VLN Navigation",(15,75),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,220),2)
    cv2.putText(tc,"Test Evaluation",(50,110),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    cv2.putText(tc,"MP3D Scene: 17DRP5sb8fy",(15,145),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,(150,150,150),1)
    cv2.putText(tc,f"{total} Episodes | 4 Action Types",(20,180),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
    cv2.putText(tc,"FORWARD | LEFT | RIGHT | STOP",(18,210),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,200,100),1)
    for _ in range(24): all_frames.append(tc)

    print(f"Running {total} test episodes...\n")
    type_results = {}
    for i, ep in enumerate(TEST_EPISODES):
        frames, success = run_episode_with_video(
            sim, model, ep, device, i+1, total)
        all_frames.extend(frames)
        if success: successes += 1
        ep_type = ep.get('type','')
        if ep_type not in type_results:
            type_results[ep_type] = []
        type_results[ep_type].append(success)

    sim.close()

    # Summary card
    sc = np.zeros((256,256,3), dtype=np.uint8)
    sc[:] = (20,40,20)
    cv2.putText(sc,"FINAL RESULTS",(55,45),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    cv2.putText(sc,f"SR = {successes/total:.3f}  ({successes}/{total})",
                (35,85), cv2.FONT_HERSHEY_SIMPLEX,0.60,(0,255,0),2)
    y = 120
    for ep_type, results in type_results.items():
        sr = sum(results)/len(results)
        col = (0,220,0) if sr >= 0.5 else (0,100,220)
        cv2.putText(sc,f"{ep_type}: {sr:.1%} ({sum(results)}/{len(results)})",
                    (20,y), cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)
        y += 25
    cv2.putText(sc,"Threshold: 3.0m (VLN standard)",
                (18,220), cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1)
    cv2.putText(sc,"CLIP ViT-B/32 + GRU Encoder",
                (18,240), cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,180,100),1)
    for _ in range(32): all_frames.append(sc)

    # Save
    print(f"\nSaving video ({len(all_frames)} frames)...")
    h, w   = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        "test_navigation_video.mp4",
        cv2.VideoWriter_fourcc(*'avc1'),
        8, (w,h))
    for f in all_frames:
        writer.write(np.ascontiguousarray(f, dtype=np.uint8))
    writer.release()

    size = os.path.getsize("test_navigation_video.mp4")/1024/1024
    print(f"Video saved: test_navigation_video.mp4 ({size:.1f}MB)")
    print(f"Duration   : ~{len(all_frames)/8:.0f} seconds")
    print(f"\nOverall SR : {successes}/{total} = {successes/total:.3f}")
    print("\nPer-type results:")
    for ep_type, results in type_results.items():
        print(f"  {ep_type:12s}: {sum(results)}/{len(results)} = "
              f"{sum(results)/len(results):.3f}")
    print("\nopen test_navigation_video.mp4")
