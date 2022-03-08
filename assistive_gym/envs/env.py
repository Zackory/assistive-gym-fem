import os, time, configparser, math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from screeninfo import get_monitors
import pybullet as p
from keras.models import load_model

from .util import Util
from .human_creation import HumanCreation
from .agents import agent, human, robot, panda, tool, furniture
from .agents.agent import Agent
from .agents.human import Human
from .agents.robot import Robot
from .agents.panda import Panda
from .agents.tool import Tool
from .agents.furniture import Furniture

class AssistiveEnv(gym.Env):
    def __init__(self, robot=None, human=None, task='', obs_robot_len=0, obs_human_len=0, time_step=0.02, frame_skip=5, render=False, gravity=-9.81, seed=1001, deformable=False):
        self.task = task
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gravity = gravity
        self.id = None
        self.gui = False
        self.gpu = False
        self.view_matrix = None
        self.deformable = deformable
        #!! CHANGED FOR BEDDING MANIPULATION TASK (CMAES!)!! GO BACK AND CHANGE LATER
        # self.seed(seeding.create_seed())
        self.seed(seed)
        if render:
            self.render()
        else:
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)

        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.human_creation = HumanCreation(self.id, np_random=self.np_random, cloth=('dressing' in task))
        self.human_limits_model = load_model(os.path.join(self.directory, 'realistic_arm_limits_model.h5'))
        #!! CHANGED FOR BEDDING MANIPULATION TASK!! GO BACK AND CHANGE LATER
        self.action_robot_len = 4
        self.action_human_len = len(human.controllable_joint_indices) if human is not None and human.controllable else 0
        self.action_space = spaces.Box(low=np.array([-1.0]*(self.action_robot_len+self.action_human_len), dtype=np.float32), high=np.array([1.0]*(self.action_robot_len+self.action_human_len), dtype=np.float32), dtype=np.float32)
        self.obs_robot_len = obs_robot_len
        self.obs_human_len = obs_human_len if human is not None and human.controllable else 0
        self.observation_space = spaces.Box(low=np.array([-1000000000.0]*(self.obs_robot_len+self.obs_human_len), dtype=np.float32), high=np.array([1000000000.0]*(self.obs_robot_len+self.obs_human_len), dtype=np.float32), dtype=np.float32)
        self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
        self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
        self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
        self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)

        self.agents = []
        self.plane = Agent()
        self.robot = robot
        self.human = human
        self.tool = Tool()
        self.furniture = Furniture()

        self.configp = configparser.ConfigParser()
        self.configp.read(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.ini'))
        # Human preference weights
        self.C_v = self.config('velocity_weight', 'human_preferences')
        self.C_f = self.config('force_nontarget_weight', 'human_preferences')
        self.C_hf = self.config('high_forces_weight', 'human_preferences')
        self.C_fd = self.config('food_hit_weight', 'human_preferences')
        self.C_fdv = self.config('food_velocities_weight', 'human_preferences')
        self.C_d = self.config('dressing_force_weight', 'human_preferences')
        self.C_p = self.config('high_pressures_weight', 'human_preferences')

    def step(self, action):
        raise NotImplementedError('Implement observations')

    def _get_obs(self, agent=None):
        raise NotImplementedError('Implement observations')

    def config(self, tag, section=None):
        return float(self.configp[self.task if section is None else section][tag])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_seed(self, seed=1000):
        self.np_random.seed(seed)

    def enable_gpu_rendering(self):
        self.gpu = True

    def disconnect(self):
        p.disconnect(self.id)

    def reset(self):
        # if not self.gui:
        #     # Reconnect the physics engine to forcefully clear memory when running long training scripts
        #     self.disconnect()
        #     self.id = p.connect(p.DIRECT)
        #     self.util = Util(self.id, self.np_random)
        if self.deformable:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=self.id)
        else:
            p.resetSimulation(physicsClientId=self.id)
        if self.gpu:
            self.util.enable_gpu()
        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.agents = []
        self.last_sim_time = None
        self.iteration = 0
        self.forces = []
        self.task_success = 0

    def build_assistive_env(self, furniture_type=None, fixed_human_base=True, human_impairment='random', gender='random', mass=None, body_shape=None):
        # Build plane, furniture, robot, human, etc. (just like world creation)
        # Load the ground plane
        plane = p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)
        self.plane.init(plane, self.id, self.np_random, indices=-1)
        # Randomly set friction of the ground
        self.plane.set_frictions(self.plane.base, lateral_friction=self.np_random.uniform(0.025, 0.5), spinning_friction=0, rolling_friction=0)
        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        # Create robot
        if self.robot is not None:
            self.robot.init(self.directory, self.id, self.np_random, fixed_base=not self.robot.mobile)
            self.agents.append(self.robot)
        # Create human
        if self.human is not None and isinstance(self.human, Human):
            self.human.init(self.human_creation, self.human_limits_model, fixed_human_base, human_impairment, gender, self.config, self.id, self.np_random, mass=mass, body_shape=body_shape)
            if self.human.controllable or self.human.impairment == 'tremor':
                self.agents.append(self.human)
        # Create furniture (wheelchair, bed, or table)
        if furniture_type is not None:
            self.furniture.init(furniture_type, self.directory, self.id, self.np_random, wheelchair_mounted=self.robot.wheelchair_mounted if self.robot is not None else False)

    def init_env_variables(self, reset=False):
        if len(self.action_space.low) <= 1 or reset:
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            self.update_action_space()
            # Define action/obs lengths
            self.action_robot_len = len(self.robot.controllable_joint_indices)
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))
            self.obs_human_len = len(self._get_obs('human'))
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)

    def update_action_space(self):
        action_len = np.sum([len(a.controllable_joint_indices) for a in self.agents if not isinstance(a, Human) or a.controllable])
        self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)

    def create_human(self, controllable=False, controllable_joint_indices=[], fixed_base=False, human_impairment='random', gender='random', mass=None, radius_scale=1.0, height_scale=1.0):
        '''
        human_impairement in ['none', 'limits', 'weakness', 'tremor']
        gender in ['male', 'female']
        '''
        self.human = Human(controllable_joint_indices, controllable=controllable)
        self.human.init(self.human_creation, self.human_limits_model, fixed_base, human_impairment, gender, None, self.id, self.np_random, mass=mass, radius_scale=radius_scale, height_scale=height_scale)
        if controllable or self.human.impairment == 'tremor':
            self.agents.append(self.human)
            self.update_action_space()
        return self.human

    def create_robot(self, robot_class, controllable_joints='right', fixed_base=True):
        self.robot = robot_class(controllable_joints)
        self.robot.init(self.directory, self.id, self.np_random, fixed_base=fixed_base)
        self.agents.append(self.robot)
        self.update_action_space()
        return self.robot

    def take_step(self, actions, gains=None, forces=None, action_multiplier=0.05, step_sim=True, ik=False):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple):
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        self.iteration += 1
        self.forces = []
        actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
        actions *= action_multiplier
        action_index = 0
        for i, agent in enumerate(self.agents):
            needs_action = not isinstance(agent, Human) or agent.controllable
            if needs_action:
                if not ik or isinstance(agent, Human):
                    agent_action_len = len(agent.controllable_joint_indices)
                else:
                    agent_action_len = 3 + 4 # 3 positon, 4 quaternion
                action = np.copy(actions[action_index:action_index+agent_action_len])
                action_index += agent_action_len
                if isinstance(agent, Robot):
                    action *= agent.action_multiplier
                if len(action) != agent_action_len:
                    print('Received agent actions of length %d does not match expected action length of %d' % (len(action), agent_action_len))
                    exit()
            # Append the new action to the current measured joint angles
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            if not ik or isinstance(agent, Human):
                # Update the target robot/human joint angles based on the proposed action and joint limits
                for _ in range(self.frame_skip):
                    if needs_action:
                        below_lower_limits = agent_joint_angles + action < agent.controllable_joint_lower_limits
                        above_upper_limits = agent_joint_angles + action > agent.controllable_joint_upper_limits
                        action[below_lower_limits] = 0
                        action[above_upper_limits] = 0
                        agent_joint_angles[below_lower_limits] = agent.controllable_joint_lower_limits[below_lower_limits]
                        agent_joint_angles[above_upper_limits] = agent.controllable_joint_upper_limits[above_upper_limits]
                    if isinstance(agent, Human) and agent.impairment == 'tremor':
                        if needs_action:
                            agent.target_joint_angles += action
                        agent_joint_angles = agent.target_joint_angles + agent.tremors * (1 if self.iteration % 2 == 0 else -1)
                    else:
                        agent_joint_angles += action
            else:
                joint = agent.right_end_effector if 'right' in agent.controllable_joints else agent.left_end_effector
                ik_indices = agent.right_arm_ik_indices if 'right' in agent.controllable_joints else agent.left_arm_ik_indices
                # NOTE: Adding action to current pose can cause drift over time
                pos, orient = agent.get_pos_orient(joint)
                # NOTE: Adding action to target pose can cause large targets far outside of the robot's work space that take a long time to come back from
                # pos, orient = np.copy(agent.target_ee_position), np.copy(agent.target_ee_orientation)
                # print('Reached pos:', pos, 'Reached orient:', orient)
                # print('Reached pos:', pos, 'Reached orient:', self.get_euler(orient))
                pos += action[:len(pos)]
                orient += action[len(pos):]
                # orient = self.get_quaternion(self.get_euler(orient) + action[len(pos):len(pos)+3]) # NOTE: RPY
                # print('Target pos:', pos, 'Target orient:', orient)
                # print('Target pos:', pos, 'Target orient:', self.get_euler(orient) + action[len(pos):len(pos)+3])
                agent_joint_angles = agent.ik(joint, pos, orient, ik_indices, max_iterations=200, use_current_as_rest=True)
            if isinstance(agent, Robot) and agent.action_duplication is not None:
                agent_joint_angles = np.concatenate([[a]*d for a, d in zip(agent_joint_angles, self.robot.action_duplication)])
                agent.control(agent.all_controllable_joints, agent_joint_angles, agent.gains, agent.forces)
            else:
                agent.control(agent.controllable_joint_indices, agent_joint_angles, gains[i], forces[i])
        if step_sim:
            # Update all agent positions
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                for agent in self.agents:
                    if isinstance(agent, Human):
                        agent.enforce_joint_limits()
                        if agent.controllable:
                            agent.enforce_realistic_joint_limits()
                self.update_targets()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()

    def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        # --- Scratching, Wiping ---
        # Any force away from target is low
        reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

        # --- Scooping, Feeding, Drinking ---
        if self.task in ['feeding', 'drinking']:
            # Penalty when robot's body applies force onto a person
            reward_force_nontarget = -total_force_on_human
        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)

        # --- Dressing ---
        # Penalty when cloth applies force onto a person
        reward_dressing_force = -np.sum(np.linalg.norm(dressing_forces, axis=-1))

        # --- Arm Manipulation ---
        # Penalty for applying large pressure to the person (high forces over small surface areas)
        if self.task == 'arm_manipulation':
            tool_right_contact_points = len(self.tool_right.get_closest_points(self.human, distance=0.01)[-1])
            tool_left_contact_points = len(self.tool_left.get_closest_points(self.human, distance=0.01)[-1])
            tool_right_pressure = 0 if tool_right_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[0] / tool_right_contact_points)
            tool_left_pressure = 0 if tool_left_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[1] / tool_left_contact_points)

            reward_arm_manipulation_tool_pressures = -(tool_right_pressure + tool_left_pressure)
            reward_force_nontarget = -(arm_manipulation_total_force_on_human - np.sum(arm_manipulation_tool_forces_on_human))
        else:
            reward_arm_manipulation_tool_pressures = 0.0

        return self.C_v*reward_velocity + self.C_f*reward_force_nontarget + self.C_hf*reward_high_target_forces + self.C_fd*reward_food_hit_human + self.C_fdv*reward_food_velocities + self.C_d*reward_dressing_force + self.C_p*reward_arm_manipulation_tool_pressures

    def init_robot_pose(self, target_ee_pos, target_ee_orient, start_pos_orient, target_pos_orients, arm='right', tools=[], collision_objects=[], wheelchair_enabled=True, right_side=True, max_iterations=3):
        base_position = None
        if self.robot.skip_pose_optimization:
            return base_position
        # Continually resample initial robot pose until we find one where the robot isn't colliding with the person
        for _ in range(max_iterations):
            if self.robot.mobile:
                # Randomize robot base pose
                pos = np.array(self.robot.toc_base_pos_offset[self.task])
                pos[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
                orient = np.array(self.robot.toc_ee_orient_rpy[self.task])
                if self.task != 'dressing':
                    orient[2] += self.np_random.uniform(-np.deg2rad(30), np.deg2rad(30))
                else:
                    orient = orient[0]
                self.robot.set_base_pos_orient(pos, orient)
                # Randomize starting joint angles
                self.robot.randomize_init_joint_angles(self.task)
            elif self.robot.wheelchair_mounted and wheelchair_enabled:
                # Use IK to find starting joint angles for mounted robots
                self.robot.ik_random_restarts(right=(arm == 'right'), target_pos=target_ee_pos, target_orient=target_ee_orient, max_iterations=1000, max_ik_random_restarts=1000, success_threshold=0.01, step_sim=False, check_env_collisions=False, randomize_limits=True, collision_objects=collision_objects)
            else:
                # Use TOC with JLWKI to find an optimal base position for the robot near the person
                base_position, _, _ = self.robot.position_robot_toc(self.task, arm, start_pos_orient, target_pos_orients, self.human, step_sim=False, check_env_collisions=False, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
            # Check if the robot or tool is colliding with objects in the environment. If so, then continue sampling.
            dists_list = []
            for tool in tools:
                tool.reset_pos_orient()
                for obj in collision_objects:
                    dists_list.append(tool.get_closest_points(obj, distance=0)[-1])
            for obj in collision_objects:
                dists_list.append(self.robot.get_closest_points(obj, distance=0)[-1])
            if all(not d for d in dists_list):
                break
        return base_position

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def update_targets(self):
        pass

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            if self.id is not None:
                self.disconnect()
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except Exception as e:
                self.width = 1920
                self.height = 1080
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))
            self.util = Util(self.id, self.np_random)

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def setup_camera_rpy(self, camera_target=[-0.2, 0, 0.75], distance=1.5, rpy=[0, -35, 40], fov=60, camera_width=1920//4, camera_height=1080//4):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def get_camera_image_depth(self, light_pos=[0, -3, 1], shadow=False, ambient=0.8, diffuse=0.3, specular=0.1):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, img, depth, _ = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, lightDirection=light_pos, shadow=shadow, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))
        depth = np.reshape(depth, (h, w))
        return img, depth
    
    def getRayFromTo(self, mouseX, mouseY):
        width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera(
        )

        # print('get ray from to')
        # print(width, height)
        # print(camForward)
        # print(horizon)
        # print(vertical)
        # print(dist)
        # print(camTarget)
        
        camPos = [
            camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
            camTarget[2] - dist * camForward[2]
        ]
        farPlane = 10000
        rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
        lenFwd = math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] +
                            rayForward[2] * rayForward[2])
        invLen = farPlane * 1. / lenFwd
        rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
        rayFrom = camPos
        oneOverWidth = float(1) / float(width)
        oneOverHeight = float(1) / float(height)

        dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
        dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
        rayToCenter = [
            rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
        ]
        ortho = [
            -0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] - float(mouseY) * dVer[0],
            -0.5 * horizon[1] + 0.5 * vertical[1] + float(mouseX) * dHor[1] - float(mouseY) * dVer[1],
            -0.5 * horizon[2] + 0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
        ]

        rayTo = [
            rayFrom[0] + rayForward[0] + ortho[0], rayFrom[1] + rayForward[1] + ortho[1],
            rayFrom[2] + rayForward[2] + ortho[2]
        ]
        lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2])
        alpha = math.atan(lenOrtho / farPlane)
        return rayFrom, rayTo, alpha
    
    def get_point_cloud(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # interesting quirk, pitch cannot = -90 due to Euler angle singularity (https://github.com/bulletphysics/bullet3/issues/2194)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0], physicsClientId=self.id)

        width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, yaw, pitch, dist, camTarget = p.getDebugVisualizerCamera(
        )
        # print('1 layer')
        # print(width, height)
        # print(camForward)
        # print(horizon)
        # print(vertical)
        # print(dist)
        # print(camTarget)

        camPos = [
            camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
            camTarget[2] - dist * camForward[2]
        ]
        farPlane = 10000
        rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
        lenFwd = math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] +
                        rayForward[2] * rayForward[2])
        oneOverWidth = float(1) / float(width)
        oneOverHeight = float(1) / float(height)
        dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
        dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]

        lendHor = math.sqrt(dHor[0] * dHor[0] + dHor[1] * dHor[1] + dHor[2] * dHor[2])
        lendVer = math.sqrt(dVer[0] * dVer[0] + dVer[1] * dVer[1] + dVer[2] * dVer[2])

        cornersX = [0, width, width, 0]
        cornersY = [0, 0, height, height]
        corners3D = []

        imgW = int(width / 10)
        imgH = int(height / 10)


        img = p.getCameraImage(imgW, imgH, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        print(img)
        rgbBuffer = np.reshape(img[2], (imgH, imgW, 4))
        # NOTE: this depth buffer's reshaping does not match the [w, h] convention for
        # OpenGL depth buffers.  See getCameraImageTest.py for an OpenGL depth buffer
        depthBuffer = np.reshape(img[3], [imgH, imgW])
        # print("rgbBuffer.shape=", rgbBuffer.shape)
        # print("depthBuffer.shape=", depthBuffer.shape)

        # disable rendering temporary makes adding objects faster
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        # visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 1, 1, 1], radius=0.01)
        # collisionShapeId = -1  #p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="duck_vhacd.obj", collisionFramePosition=shift,meshScale=meshScale)

        for i in range(4):
            w = cornersX[i]
            h = cornersY[i]
            rayFrom, rayTo, _ = self.getRayFromTo(w, h)
            rf = np.array(rayFrom)
            rt = np.array(rayTo)
            vec = rt - rf
            l = np.sqrt(np.dot(vec, vec))
            newTo = (0.01 / l) * vec + rf
            #print("len vec=",np.sqrt(np.dot(vec,vec)))

            # p.addUserDebugLine(rayFrom, newTo, [1, 0, 0])
            corners3D.append(newTo)
        count = 0

        stepX = 2
        stepY = 2


        point_cloud = []
        point_cloud_filtered = []  # * not used for data collection
        for w in range(0, imgW, stepX):
            for h in range(0, imgH, stepY):
                count += 1
                # if ((count % 100) == 0):
                #     print(count, "out of ", imgW * imgH / (stepX * stepY))
                rayFrom, rayTo, alpha = self.getRayFromTo(w * (width / imgW), h * (height / imgH))
                rf = np.array(rayFrom)
                rt = np.array(rayTo)
                vec = rt - rf
                l = np.sqrt(np.dot(vec, vec))
                depthImg = float(depthBuffer[h, w])
                far = 1000.
                near = 0.01
                depth = far * near / (far - (far - near) * depthImg)
                depth /= math.cos(alpha)
                newTo = (depth / l) * vec + rf
                point_cloud.append(newTo)
                # p.addUserDebugLine(rayFrom, newTo, [1, 0, 0])

                # * don't filter point cloud for bu_gnn_naive, return unfiltered
                # * z pos must be greater than height of the bed
                # if (newTo[2] > 0.585) and (abs(newTo[1])<1.05):
                #     point_cloud_filtered.append(newTo)
                #     mb = p.createMultiBody(baseMass=0,
                #                         baseCollisionShapeIndex=collisionShapeId,
                #                         baseVisualShapeIndex=visualShapeId,
                #                         basePosition=newTo,
                #                         useMaximalCoordinates=True)
                #     color = rgbBuffer[h, w]
                #     color = [color[0] / 255., color[1] / 255., color[2] / 255., 1]
                #     p.changeVisualShape(mb, -1, rgbaColor=color)

        # p.addUserDebugLine(corners3D[0], corners3D[1], [1, 0, 0])
        # p.addUserDebugLine(corners3D[1], corners3D[2], [1, 0, 0])
        # p.addUserDebugLine(corners3D[2], corners3D[3], [1, 0, 0])
        # p.addUserDebugLine(corners3D[3], corners3D[0], [1, 0, 0])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # print("ready\n")
        print(len(point_cloud), len(point_cloud_filtered))
        
        # import pickle
        # f = open('point_cloud_data.pkl', 'wb')

        # # width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget
        # point_cloud_info = {
        #     'width':width,
        #     'height':height,
        #     'viewMat':viewMat,
        #     'projMat':projMat,
        #     'cameraUp':cameraUp,
        #     'camForward':camForward,
        #     'horizon':horizon,
        #     'vertical':vertical,
        #     'dist':dist,
        #     'camTarget':camTarget,
        #     'imgW':imgW,
        #     'imgH':imgW,
        #     'w':w,
        #     'h':h,
        #     'yaw':yaw,
        #     'pitch':pitch
        # }
        # pickle.dump(point_cloud_info, f)

        return point_cloud
    
    def get_depth_image_for_point_cloud(self, image_width, image_height, view_matrix, projection_matrix):
        # p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, lightDirection=light_pos, shadow=shadow, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.id)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0], physicsClientId=self.id)
        img = p.getCameraImage(image_width, image_height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return img

    def create_box(self, half_extents=[1, 1, 1], mass=0.0, pos=[0, 0, 0], orientation=[0, 0, 0, 1], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        box_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.id) if collision else -1
        box_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return box_collision, box_visual
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=box_collision, baseVisualShapeIndex=box_visual, basePosition=pos, baseOrientation=orientation, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        box = Agent()
        box.init(body, self.id, self.np_random, indices=-1)
        return box

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return sphere_collision, sphere_visual
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=pos, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        sphere = Agent()
        sphere.init(body, self.id, self.np_random, indices=-1)
        return sphere

    def create_spheres(self, radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]], visual=True, collision=True, rgba=[0, 1, 1, 1]):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        sphere_ids = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
        spheres = []
        for body in sphere_ids:
            sphere = Agent()
            sphere.init(body, self.id, self.np_random, indices=-1)
            spheres.append(sphere)
        return spheres

    def create_agent_from_obj(self, visual_filename, collision_filename, scale=1.0, mass=1.0, pos=[0, 0, 0], orient=[0, 0, 0, 1], rgba=[1, 1, 1, 1], maximal=False):
        visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=scale, rgbaColor=rgba, physicsClientId=self.id)
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=scale, physicsClientId=self.id)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=pos, baseOrientation=orient, useMaximalCoordinates=maximal, physicsClientId=self.id)
        agent = Agent()
        agent.init(body, self.id, self.np_random, indices=-1)
        return agent
    
    def create_capsule(self, radius=0, length=0, position = [2,0,1], orientation = [0, 1, 1, 1], maximal_coordinates=False):
        collision_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=length, physicsClientId=self.id)
        visual_shape = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length, rgbaColor=[0,0,0,1],physicsClientId=self.id)
        mass = 0
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=position, baseOrientation = orientation, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        capsule = Agent()
        capsule.init(body, self.id, self.np_random, indices=-1)
        return capsule

