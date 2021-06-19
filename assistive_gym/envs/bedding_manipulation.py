import os, time
import numpy as np
from numpy.lib.function_base import append
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

# python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --algo ppo --train --train-timesteps 1000 --save-dir ./trained_models/
# python3 -m assistive_gym.learn --env "BeddingManipulationSphere-v1" --algo ppo --render --seed 0 --load-policy-path ./trained_models/ --render-episodes 1


class BeddingManipulationEnv(AssistiveEnv):
    def __init__(self, robot, human, use_mesh=False):
        if robot is None:
            super(BeddingManipulationEnv, self).__init__(robot=None, human=human, task='bedding_manipulation', obs_robot_len=1, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True)
            self.use_mesh = use_mesh

    def step(self, action):
        obs = self._get_obs()

        # scale bounds the 2D grasp and release locations to the area over the mattress (action nums only in range [-1, 1])
        scale = [0.44, 1.05]
        grasp_loc = action[0:2]*scale
        release_loc = action[2:]*scale
        # print(grasp_loc, release_loc)


        # move sphere to 2D grasp location, some arbitrary distance z = 1 in the air
        #! don't technically need to do this, remove later
        # self.sphere_ee.set_base_pos_orient(np.append(grasp_loc, 1), np.array([0,0,0]))

        # get points on the blanket, initial state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # print("got blanket data")

        # reward_uncover_target = self.uncover_target_reward(data)
        # print("target reward: ", reward_uncover_target)
        # reward_uncover_nontarget = self.uncover_nontarget_reward(data)
        # print("non target reward: ", reward_uncover_nontarget)

        # calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist = []
        for i, v in enumerate(data[1]):
            v = np.array(v)
            d = np.linalg.norm(v[0:2] - grasp_loc)
            dist.append(d)
        anchor_idx = np.argpartition(np.array(dist), 4)[:4]
        # for a in anchor_idx:
            # print("anchor loc: ", data[1][a])

        # update grasp_loc var with the location of the central anchor point on the cloth
        grasp_loc = np.array(data[1][anchor_idx[0]][0:2])
        # print("GRASP LOC =", grasp_loc)

        # move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
        self.sphere_ee.set_base_pos_orient(data[1][anchor_idx[0]], np.array([0,0,0]))
        constraint_ids = []
        constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))

        for i in anchor_idx[1:]:
            pos_diff = np.array(data[1][i]) - np.array(data[1][anchor_idx[0]])
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, [0, 0, 0]))
        # print("sphere moved to grasp loc, anchored")


        # move sphere up to the arbitrary z position z = 1
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        # print(current_pos[2])
        delta_z = 0.5                           # distance to move up
        final_z = delta_z + current_pos[2]      # global z position after moving up delta z
        while current_pos[2] <= final_z:
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            # print(current_pos[2])

        # print(f"sphere moved {delta_z}, current z pos {current_pos[2]}")


        # move sphere to the release location, release the blanket
        travel_dist = release_loc - grasp_loc

        # determine delta x and y, make sure it is, at max, close to 0.005
        num_steps = np.abs(travel_dist//0.005).max()
        delta_x, delta_y = travel_dist/num_steps

        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        for _ in range(int(num_steps)):
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, 0]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            # print("current: ", current_pos, " release: ", release_loc)
        # print("sphere moved to release loc, blanket settling")

        # continue stepping simulation to allow the cloth to settle before release
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.id)
        # print("release cloth, allow to settle")


        # release the cloth at the release point, sphere is at the same arbitrary z position in the air
        for i in constraint_ids:
            p.removeConstraint(i, physicsClientId=self.id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)
        # print("done")


        # get points on the blanket, final state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)


        reward_uncover_target = self.uncover_target_reward(data)
        reward_uncover_nontarget = self.uncover_nontarget_reward(data)
        reward_distance_btw_grasp_release = -150 if np.linalg.norm(grasp_loc - release_loc) >= 1.5 else 0


        # print(reward_uncover_target, reward_uncover_nontarget, reward_distance_btw_grasp_release)
        reward = self.config('uncover_target_weight')*reward_uncover_target + self.config('uncover_nontarget_weight')*reward_uncover_nontarget + self.config('grasp_release_distance_max_weight')*reward_distance_btw_grasp_release
        
        # print("reward: ", reward)


        info = {}
        self.iteration += 1
        done = self.iteration >= 1

        # time.sleep(1)

        # return 0, 0, 1, {}
        return obs, reward, done, info

    def change_point_color(self, points_target_limb, limb, ind, rgb = [0, 1, 0.5, 1]):
        p.changeVisualShape(points_target_limb[limb][ind].body, -1, rgbaColor=rgb, flags=0, physicsClientId=self.id)


    def uncover_target_reward(self, blanket_state):
        points_covered = 0
        uncovered_rgb = [0, 1, 0.5, 1]
        covered_rgb = [1, 1, 1, 1]
        threshold = 0.028
        total_points = self.total_target_point_count

        # count number of target points covered by the blanket
        for limb, points_pos_target_limb_world in self.points_pos_target_limb_world.items():
            for point in range(len(points_pos_target_limb_world)):
                covered = False
                for i, v in enumerate(blanket_state[1]):
                    # target_foot = np.array(target_foot)
                    # v = np.array(v)
                    if abs(np.linalg.norm(v[0:2]-points_pos_target_limb_world[point][0:2])) < threshold:
                        covered = True
                        points_covered += 1
                        break
                # rgb = covered_rgb if covered else uncovered_rgb
                # self.change_point_color(self.points_target_limb, limb, point, rgb = rgb)

        points_uncovered = total_points - points_covered

        # print("total_targets:", total_points)
        # print("uncovered", points_uncovered)

        return (points_uncovered/total_points)*100

    def uncover_nontarget_reward(self, blanket_state):
        points_covered = 0
        uncovered_rgb = [1, 0, 0, 1]
        covered_rgb = [0, 0, 1, 1]
        threshold = 0.028
        total_points = self.total_nontarget_point_count

        # count number of target points covered by the blanket
        for limb, points_pos_nontarget_limb_world in self.points_pos_nontarget_limb_world.items():
            for point in range(len(points_pos_nontarget_limb_world)):
                covered = False
                for i, v in enumerate(blanket_state[1]):
                    # target_foot = np.array(target_foot)
                    # v = np.array(v)
                    if abs(np.linalg.norm(v[0:2]-points_pos_nontarget_limb_world[point][0:2])) < threshold:
                        covered = True
                        points_covered += 1
                        break
                # rgb = covered_rgb if covered else uncovered_rgb
                # self.change_point_color(self.points_nontarget_limb, limb, point, rgb = rgb)
        points_uncovered = total_points - points_covered

        # print("total_targets:", total_points)
        # print("uncovered", points_uncovered)

        # 100 when all points uncovered, 0 when all still covered
        return (points_uncovered/total_points)*-100
        

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(BeddingManipulationEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed', body_shape=np.zeros((1, 10)))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        
        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, -0.2, 0.8], [-np.pi/2.0, 0, np.pi])

        # time.sleep(10)
        # Seperate the human's legs so that it's easier to uncover a single shin
        current_l = self.human.get_joint_angles(self.human.left_leg_joints)
        current_l[1] = -0.2
        current_r = self.human.get_joint_angles(self.human.right_leg_joints)
        current_r[1] = 0.2
        self.human.set_joint_angles(self.human.left_leg_joints, current_l, use_limits=True, velocities=0)
        self.human.set_joint_angles(self.human.right_leg_joints, current_r, use_limits=True, velocities=0)

        # set shoulder angle so that person's pose is the same for each rollout
        current_l = self.human.get_joint_angles(self.human.left_arm_joints)
        current_l[3] = -0.2
        current_r = self.human.get_joint_angles(self.human.right_arm_joints)
        current_r[3] = 0.2
        self.human.set_joint_angles(self.human.left_arm_joints, current_l, use_limits=True, velocities=0)
        self.human.set_joint_angles(self.human.right_arm_joints, current_r, use_limits=True, velocities=0)
        

        # time.sleep(2)
        # Let the person settle on the bed
        p.setGravity(0, 0.2, -2, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)


        # Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        

        # time.sleep(600)

        if self.use_mesh:
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -60), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])


        shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        
        # can index possible targets differently to uncover different target limbs, current set to shin and foot
        self.target_limbs = self.human.all_possible_target_limbs[4]

        self.generate_points_along_body()
        # time.sleep(600)


        # spawn blanket
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        #TODO Adjust friction, should be lower so that the cloth can slide over the limbs
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)

        # change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # Drop the blanket on the person, allow to settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)
    
        if self.robot is None:
            position = np.array([-0.3, -0.86, 0.8])
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])
            # self.robot = self.sphere_ee


        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # vertex_index = 480
        # new_vertex_position = np.array(data[1][vertex_index])
        # p.createSoftBodyAnchor(self.blanket, vertex_index, self.sphere_ee.body, -1, [0, 0, 0])


        # # Add vertices
        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # for i, v in enumerate(data[1]):
        #     p.addUserDebugText(text=str(i), textPosition=v, textColorRGB=[0, 0, 0], textSize=1, lifeTime=0, physicsClientId=self.id)
        # import time
        # time.sleep(3000)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Initialize enviornment variables
        # self.time = time.time()
        if self.robot is None:      # Sphere manipulator
            from gym import spaces
            # modified version of init_env_variables
            # update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 4
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)

            # Define action/obs lengths
            self.action_robot_len = 4
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))    # 1
            self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)

        else:
            self.init_env_variables()

        # time.sleep(60)
            
        return self._get_obs()


    def generate_points_along_body(self):

        self.point_indices_to_ignore = []

        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}
        self.total_target_point_count = 0

        self.points_pos_on_nontarget_limb = {}
        self.points_nontarget_limb = {}
        self.total_nontarget_point_count = 0

        #TODO: NEED TO DEAL WITH THE HANDS (sphere)
        for limb in self.human.all_body_parts:
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]
            if limb in [18, 28]:
                pass
            #! COULD JUST REMOVE HANDS?? LOOK INTO WHEN WORKING ON TORSO
            elif limb in self.target_limbs:
                self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[1, 1, 1, 1])
                self.total_target_point_count += len(self.points_pos_on_target_limb[limb])
            else:
                # print("nontarget limb")
                self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                self.points_nontarget_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_nontarget_limb[limb]), visual=True, collision=False, rgba=[0, 0, 1, 1])
                self.total_nontarget_point_count += len(self.points_pos_on_nontarget_limb[limb])

        self.update_points_along_body()
    
    def update_points_along_body(self):

        self.points_pos_target_limb_world = {}
        self.points_pos_nontarget_limb_world = {}
        for limb in self.human.all_body_parts:
            # get current position and orientation of the limbs, apply a correction to the pos, orient if necessary
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])
            points_pos_limb_world = []

            if limb in [18, 28]:
                pass
            elif limb in self.target_limbs:
                for points_pos_on_target_limb, point in zip(self.points_pos_on_target_limb[limb], self.points_target_limb[limb]):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, points_pos_on_target_limb, [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    point.set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_target_limb_world[limb] = points_pos_limb_world
            else:
                # print("nontarget limb")
                for points_pos_on_nontarget_limb, point in zip(self.points_pos_on_nontarget_limb[limb], self.points_nontarget_limb[limb]):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, points_pos_on_nontarget_limb, [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    point.set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_nontarget_limb_world[limb] = points_pos_limb_world


    
    

