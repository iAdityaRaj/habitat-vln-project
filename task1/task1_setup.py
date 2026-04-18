import habitat_sim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

obs = sim.get_sensor_observations()
rgb = obs["color_sensor"]

print("✅ Scene loaded!")
print("RGB shape:", rgb.shape)
print("Action space: STOP | MOVE_FORWARD | TURN_LEFT | TURN_RIGHT")

plt.imsave("initial_observation.png", rgb[:, :, :3])
print("✅ Saved: initial_observation.png")
sim.close()
