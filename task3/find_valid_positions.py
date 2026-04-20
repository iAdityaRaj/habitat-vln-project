import habitat_sim
import numpy as np

SCENE = "/Users/adityaraj/AdityaRaj/habitat_vln_project/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

sim_cfg          = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE
rgb              = habitat_sim.CameraSensorSpec()
rgb.uuid         = "color_sensor"
rgb.sensor_type  = habitat_sim.SensorType.COLOR
rgb.resolution   = [256, 256]
rgb.position     = [0.0, 1.5, 0.0]
agent_cfg        = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb]
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

agent         = sim.initialize_agent(0)
valid_actions = list(agent.agent_config.action_space.keys())
print(f"Valid actions: {valid_actions}")

# Find navigable points
print("\nFinding navigable positions...")
navigable = []
for _ in range(20):
    pt = sim.pathfinder.get_random_navigable_point()
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    navigable.append([x, y, z])
    print(f"  [{x:.3f}, {y:.3f}, {z:.3f}]")

print(f"\nTesting movement from first point...")
state          = habitat_sim.AgentState()
state.position = np.array(navigable[0])
agent.set_state(state)

pos_before = agent.get_state().position
print(f"Before: [{pos_before[0]:.3f}, {pos_before[1]:.3f}, {pos_before[2]:.3f}]")

sim.step('move_forward')
pos_after = agent.get_state().position
print(f"After : [{pos_after[0]:.3f}, {pos_after[1]:.3f}, {pos_after[2]:.3f}]")

import math
dist = math.sqrt(sum((float(pos_after[i])-float(pos_before[i]))**2 for i in range(3)))
print(f"Moved : {dist:.4f}m")

sim.close()
