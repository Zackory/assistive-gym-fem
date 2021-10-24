from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene

import assistive_gym, colorsys
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.stretch import Stretch
from assistive_gym.envs.agents.agent import Agent
import assistive_gym.envs.agents.human as h
from assistive_gym.envs.agents.human_mesh import HumanMesh
from assistive_gym.envs.agents.tool import Tool

import pybullet as p
import numpy as np
import time

# Create an empty Assistive Gym environment
env = AssistiveEnv()
env.render()
env.set_seed(200)
# env.setup_camera(camera_eye=[1.5, -2, 2], camera_target=[-0.6, -0.5, 0.7], fov=60, camera_width=1920//4, camera_height=1080//4)
env.setup_camera(camera_eye=[1.75, -1.5, 2], camera_target=[-0.6, -0.25, 0.4], fov=60, camera_width=1920//4, camera_height=1080//4)
env.reset()

# Create the iGibson environment
scene = InteractiveIndoorScene('Rs_int', build_graph=True, pybullet_load_texture=True, texture_randomization=False, object_randomization=False)
scene.load()

# Change position of the bed (the 33rd object)
bed = Agent()
bed.init_env(33, env, indices=-1)
pos, orient = bed.get_base_pos_orient()
bed.set_base_pos_orient(pos + np.array([-0.4, 0, 0]), orient)
bed.set_mass(-1, 0)

# Change position of the pillow (the 32rd object)
pillow = Agent()
pillow.init_env(32, env, indices=-1)
pos, orient = pillow.get_base_pos_orient()
pillow.set_base_pos_orient(pos + np.array([0, 0, -0.15]), orient)
pillow.set_mass(-1, 0)

hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
hsv[-1] = 0.6
skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1.0]

human = HumanMesh()
joints_positions = [(human.j_right_shoulder_z, 80), (human.j_left_shoulder_z, -80)]
body_shape = np.zeros((1, 10))
gender = 'male'
human.init(env.directory, env.id, env.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, skin_color=skin_color, collision_enabled=False)
human_pos = np.array([-3.1, -1.15, 0.75])
human_orient = env.get_quaternion(np.array([-np.pi/2.0, 0, np.pi]))
human.set_base_pos_orient(human_pos, human_orient)
human.set_whole_body_frictions(lateral_friction=10, spinning_friction=10, rolling_friction=10)

# Create robot
robot = env.create_robot(Stretch, controllable_joints='wheel_right', fixed_base=False)
# robot.print_joint_info()
robot.set_base_pos_orient([-2.45, -0.6, 0.1], [0, 0, np.pi*3/2.0])
robot.set_joint_angles([3], [0.95]) # lift
robot.set_joint_angles([5, 6, 7, 8], [0.06]*4) # arm
# robot.set_joint_angles([9], [1.0]) # gripper
robot.set_gripper_open_position(robot.right_gripper_indices, robot.gripper_pos['feeding'], set_instantly=True)
robot.set_all_joints_stiffness(1)

while True:
    env.take_step(np.zeros(len(env.robot.controllable_joint_indices)))

# done = False
# for _ in range(60):
#     action = np.zeros(len(env.robot.controllable_joint_indices))
#     action[:len(env.robot.wheel_joint_indices)+1] = 1.0
#     # observation, reward, done, info = env.step(action)
#     env.take_step(action)
#     # img, depth = env.get_camera_image_depth()

# env.disconnect()

