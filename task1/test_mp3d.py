import habitat_sim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCENE = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

sim_cfg          = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE

rgb             = habitat_sim.CameraSensorSpec()
rgb.uuid        = "color_sensor"
rgb.sensor_type = habitat_sim.SensorType.COLOR
rgb.resolution  = [256, 256]
rgb.position    = [0.0, 1.5, 0.0]

agent_cfg       = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb]

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
obs = sim.get_sensor_observations()
rgb = obs["color_sensor"][:,:,:3]

plt.imsave("mp3d_real.png", rgb)
print("✅ Real MP3D scene loaded!")
print(f"RGB shape: {rgb.shape}")
sim.close()
