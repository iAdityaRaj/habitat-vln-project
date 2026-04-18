import habitat_sim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

SCENE = "/Users/adityaraj/AdityaRaj/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE

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
valid_actions = list(agent.agent_config.action_space.keys())

# Collect frames + positions
frames = []
positions = []

obs = sim.get_sensor_observations()
frames.append(obs["color_sensor"][:, :, :3])
positions.append(agent.get_state().position.copy())

for step in range(8):
    action = random.choice(valid_actions)
    sim.step(action)
    obs = sim.get_sensor_observations()
    frames.append(obs["color_sensor"][:, :, :3])
    positions.append(agent.get_state().position.copy())

sim.close()

# Plot what the robot saw at each step
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(frames[i])
    p = positions[i]
    ax.set_title(f"Step {i} | pos=({p[0]:.1f}, {p[2]:.1f})", fontsize=9)
    ax.axis('off')

plt.suptitle("What the Robot Saw at Each Step", fontsize=14)
plt.tight_layout()
plt.savefig("robot_journey.png", dpi=150)
print("✅ Saved: robot_journey.png")
print("Run: open robot_journey.png")
