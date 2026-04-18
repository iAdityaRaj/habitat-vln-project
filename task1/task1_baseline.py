import habitat_sim
import random

# habitat-sim 0.3.3 exact action names
ACTIONS = {
    0: "move_forward",  # we'll handle stop manually
    1: "move_forward",
    2: "turn_left",
    3: "turn_right"
}

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"

rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "color_sensor"
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor.resolution = [256, 256]
rgb_sensor.position = [0.0, 1.5, 0.0]

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_sensor]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)

# Print available actions
print("Available actions:", list(agent.agent_config.action_space.keys()))
print("\n=== Random Baseline Agent ===\n")

# Get valid action names directly from sim
valid_actions = list(agent.agent_config.action_space.keys())
trajectory = []

for step in range(50):
    # Pick random action from what's actually available
    action_name = random.choice(valid_actions)
    trajectory.append(action_name)

    sim.step(action_name)
    pos = agent.get_state().position
    print(f"Step {step+1:02d} | {action_name:15s} | pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

print(f"\n✅ Completed 50 steps")
print(f"Total steps : 50")
print(f"Trajectory  : {trajectory}")
sim.close()
