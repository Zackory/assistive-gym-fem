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

# Change position of a chair (the 15th object)
chair = Agent()
chair.init_env(15, env, indices=-1)
pos, orient = chair.get_base_pos_orient()
chair_orient = env.get_quaternion(env.get_euler(orient) + np.array([0, 0, -0.9]))
chair.set_base_pos_orient(pos + np.array([-0.4, 0, -0.15]), chair_orient)

hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
hsv[-1] = 0.4
skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1.0]

human = HumanMesh()
joints_positions = [(human.j_right_shoulder_z, 60), (human.j_right_elbow_y, 90), (human.j_left_shoulder_z, -60), (human.j_left_elbow_y, -90), (human.j_right_hip_x, -90), (human.j_right_knee_x, 90), (human.j_left_hip_x, -90), (human.j_left_knee_x, 90)]
body_shape = np.zeros((1, 10))
gender = 'female' # 'random'
human.init(env.directory, env.id, env.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]], skin_color=skin_color, collision_enabled=False)
chair_seat_position, chair_orient = chair.get_base_pos_orient()
human_pos = chair_seat_position - human.get_vertex_positions(human.bottom_index) + np.array([0, 0.13, 0.1])
human_orient = env.get_quaternion(env.get_euler(chair_orient) + np.array([0, 0, 0]))
human.set_base_pos_orient(human_pos, human_orient)
human.set_whole_body_frictions(lateral_friction=10, spinning_friction=10, rolling_friction=10)

# Create robot
robot = env.create_robot(Stretch, controllable_joints='wheel_right', fixed_base=False)
# robot.print_joint_info()
robot.set_base_pos_orient([0.7, -0.6, 0.1], [0, 0, np.pi])
robot.set_joint_angles([3], [1.05]) # lift
robot.set_joint_angles([5, 6, 7, 8], [0.1]*4) # arm
# robot.set_joint_angles([9], [1.0]) # gripper
robot.set_gripper_open_position(robot.right_gripper_indices, robot.gripper_pos['feeding'], set_instantly=True)
robot.set_all_joints_stiffness(1)


# Initialize the tool in the robot's gripper
tool = Tool()
tool.init(robot, 'feeding', env.directory, env.id, env.np_random, right=True, mesh_scale=[0.08]*3)
tool.set_gravity(0, 0, 0)
# Generate food
spoon_pos, spoon_orient = tool.get_base_pos_orient()
food_radius = 0.005
food_mass = 0.001
batch_positions = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            batch_positions.append(np.array([i*2*food_radius-0.005, j*2*food_radius, k*2*food_radius+0.01]) + spoon_pos)
foods = env.create_spheres(radius=food_radius, mass=food_mass, batch_positions=batch_positions, visual=False, collision=True)
colors = [[60./256., 186./256., 84./256., 1], [244./256., 194./256., 13./256., 1],
          [219./256., 50./256., 54./256., 1], [72./256., 133./256., 237./256., 1]]
for i, f in enumerate(foods):
    p.changeVisualShape(f.body, -1, rgbaColor=colors[i%len(colors)], physicsClientId=env.id)


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

