import os, time
import numpy as np
from numpy.lib.function_base import append
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

class BeddingManipulationEnv(AssistiveEnv):
    def __init__(self, robot, human, use_mesh=False):
        if robot is None:
            super(BeddingManipulationEnv, self).__init__(robot=None, human=human, task='bed_bathing', obs_robot_len=4, obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)), frame_skip=1, time_step=0.01, deformable=True)
            self.use_mesh = use_mesh

    def step(self, action):
        obs = self._get_obs()


        grasp_loc = action[0:2]
        release_loc = action[2:]
        print(grasp_loc, release_loc)

        # move sphere to 2D grasp location, some arbitrary distance z = 1 in the air
        #! don't technically need to do this, remove later
        self.sphere_ee.set_base_pos_orient(np.append(grasp_loc, 1), np.array([0,0,0]))


        # get points on the blanket, initial state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        print("got blanket data")

        reward_uncover_target = self.uncover_target_reward(data)
        # reward_uncover_nontarget = self.uncover_nontarget_reward(data)

        # calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist = []
        for i, v in enumerate(data[1]):
            v = np.array(v)
            d = np.linalg.norm(v[0:2] - grasp_loc)
            dist.append(d)
        anchor_idx = np.argpartition(np.array(dist), 4)[:4]
        for a in anchor_idx:
            print("anchor loc: ", data[1][a])

        # update grasp_loc var with the location of the central anchor point on the cloth
        grasp_loc = np.array(data[1][anchor_idx[0]][0:2])
        print("GRASP LOC =", grasp_loc)

        # move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
        self.sphere_ee.set_base_pos_orient(data[1][anchor_idx[0]], np.array([0,0,0]))
        constraint_ids = []
        constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))

        for i in anchor_idx[1:]:
            pos_diff = np.array(data[1][i]) - np.array(data[1][anchor_idx[0]])
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, [0, 0, 0]))
        print("sphere moved to grasp loc, anchored")


        # move sphere up to the arbitrary z position z = 1
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        # print(current_pos[2])
        while current_pos[2] <= 1.2:
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            # print(current_pos[2])

        print("sphere moved to z=1.5 up")



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
        print("sphere moved to release loc, blanket settling")

        # continue stepping simulation to allow the cloth to settle before release
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.id)

        print("release cloth, allow to settle")
        # release the cloth at the release point, sphere is at the same arbitrary z position in the air
        for i in constraint_ids:
            p.removeConstraint(i, physicsClientId=self.id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)
        print("done")


        self.iteration += 1
        done = self.iteration >= 1

        # get points on the blanket, final state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        reward_uncover_target = self.uncover_target_reward(data)
        # reward_uncover_nontarget = self.uncover_nontarget_reward(data)

        time.sleep(600)

        reward_action = -np.linalg.norm(action)
        reward = self.config('uncover_target_weight')*reward_uncover_target + self.config('action_weight')*reward_action + self.config('uncover_nontarget_weight')*reward_uncover_nontarget


        return obs, 0, done, {}


    def uncover_target_reward(self, blanket_state):
        # get points on the blanket, final state of the cloth

        # print(self.targets_pos_foot_world)
        # print(self.targets_pos_shin_world)
        points_still_covered = 0
        threshold = 0.05

        self.target_limbs = [self.human.right_shin, self.human.right_foot]

        # count number of target points covered by the blanket
        for points_pos_target_limb_world  in self.points_pos_target_limb_world.values():
            for target_limb in points_pos_target_limb_world:
                for i, v in enumerate(blanket_state[1]):
                    # target_foot = np.array(target_foot)
                    # v = np.array(v)
                    if abs(np.linalg.norm(v[0:2]-target_limb[0:2])) < threshold:
                        # print(np.linalg.norm(v[0:2]-target_foot[0:2]))
                        # print(v[0:2]-target_foot[0:2])
                        # p.addUserDebugText(text=str(i), textPosition=v, textColorRGB=[0, 0, 0], textSize=1, lifeTime=0, physicsClientId=self.id)
                        points_still_covered += 1
                        break
        print("total_target_points", self.total_target_point_count)
        print("covered", points_still_covered)

        # count_shin = 0
        # for target_shin in self.targets_pos_shin_world:
        #     for i, v in enumerate(blanket_state[1]):
        #         if abs(np.linalg.norm(v[0:2]-target_shin[0:2])) < threshold:
        #             # p.addUserDebugText(text=str(i), textPosition=v, textColorRGB=[0, 0, 0], textSize=1, lifeTime=0, physicsClientId=self.id)
        #             count_shin += 1
        #             break

        # print("total_targets, shin", len(self.targets_pos_on_shin))
        # print("covered", count_shin)
        return points_still_covered


    #! NEED REDO!!!
    def uncover_nontarget_reward(self, blanket_state):
        counts = [0, 0, 0]
        threshold = 0.05

        # count number of nontarget points covered by the blanket
        for target_upperarm in self.targets_pos_upperarm_world:
            for i, v in enumerate(blanket_state[1]):
                if abs(np.linalg.norm(v[0:2]-target_upperarm[0:2])) < threshold:
                    counts[0] += 1
                    break

        for target_forearm in self.targets_pos_forearm_world:
            for i, v in enumerate(blanket_state[1]):
                if abs(np.linalg.norm(v[0:2]-target_forearm[0:2])) < threshold:
                    counts[1] += 1
                    break


        for target_thigh in self.targets_pos_thigh_world:
            for i, v in enumerate(blanket_state[1]):
                if abs(np.linalg.norm(v[0:2]-target_thigh[0:2])) < threshold:
                    counts[2] += 1
                    break

        print("total_targets, upperarm", len(self.targets_pos_on_upperarm))
        print("uncovered", len(self.targets_pos_on_upperarm) - counts[0])
        print("total_targets, forearm", len(self.targets_pos_on_forearm))
        print("uncovered", len(self.targets_pos_on_forearm) - counts[1])
        print("total_targets, thigh", len(self.targets_pos_on_thigh))
        print("uncovered", len(self.targets_pos_on_thigh) - counts[2])
        

        return 0

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(BeddingManipulationEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        
        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, -0.4, 0.8], [-np.pi/2.0, 0, np.pi])


        # Seperate the human's legs so that it's easier to uncover a single shin
        current_l = self.human.get_joint_angles(self.human.left_leg_joints)
        current_l[1] = -0.2
        current_r = self.human.get_joint_angles(self.human.right_leg_joints)
        current_r[1] = 0.2
        self.human.set_joint_angles(self.human.left_leg_joints, current_l, use_limits=True, velocities=0)
        self.human.set_joint_angles(self.human.right_leg_joints, current_r, use_limits=True, velocities=0)


        # Let the person settle on the bed
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)


        # Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        

        if self.use_mesh:
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -60), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])

        # r = 0.1
        # l = 0.5
        # self.capsule = self.create_capsule(r, l)
        # self.targets_pos_on_cap = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -l/2]), radius=r, distance_between_points=0.03)
        # self.targets_cap = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_cap), visual=True, collision=False, rgba=[0, 1, 1, 1])
        # cap_pos, cap_orient = self.capsule.get_base_pos_orient()
        # for target_pos_on_cap, target in zip(self.targets_pos_on_cap, self.targets_cap):
        #     # target_pos = np.array(p.multiplyTransforms(foot_pos, foot_orient, target_pos_on_foot, self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)[0])
        #     target_pos = np.array(p.multiplyTransforms(cap_pos, cap_orient, target_pos_on_cap, [0, 0, 0, 1], physicsClientId=self.id)[0])
        #     target.set_base_pos_orient(target_pos, [0, 0, 0, 1])


        shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        self.generate_points_along_body()
        # self.generate_targets()
        # self.generate_nontargets()

        # time.sleep(600)


        # spawn blanket
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=2, useFaceContact=1, physicsClientId=self.id)

        # change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.blanket, [0, 0, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)



        # Drop the blanket on the person, allow to settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)
    
        if self.robot is None:
            position = np.array([-0.3, -0.86, 0.8])
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])


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

        # Setup camera for taking depth images
        # self.setup_camera(camera_eye=[0, 0, 0.305+2.101], camera_target=[0, 0, 0.305], fov=60, camera_width=1920//4, camera_height=1080//4)
        # self.setup_camera(camera_eye=[0, 0, 0.305+0.101], camera_target=[0, 0, 0.305], fov=60, camera_width=1920//4, camera_height=1080//4)
        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 0], fov=60, camera_width=468//2, camera_height=398)
        # 468 x 398
        # self.setup_camera(camera_eye=[0.5, 0.75, 1.5], camera_target=[-0.2, 0, 0.75])
        img, depth = self.get_camera_image_depth()
        depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
        depth = (depth * 255).astype(np.uint8)
        print(depth)
        # cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
        cv2.imshow('image', depth_colormap)
        # plt.imshow(img)
        # plt.show()
        # Image.fromarray(img[:, :, :3], 'RGB').show()

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Initialize enviornment variables
        self.time = time.time()
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
            self.obs_robot_len = len(self._get_obs('robot'))    # 0
            self.obs_human_len = len(self._get_obs('human'))    # 0
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

        self.target_limbs = [self.human.right_shin, self.human.right_foot]

        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}
        self.total_target_point_count = 0
        for limb in self.target_limbs:
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]
            self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
            self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[0, 1, 1, 1])
            self.total_target_point_count += len(self.points_pos_on_target_limb[limb])


        # self.nontarget_limbs = list(set(self.human.limbs)-set(self.target_limbs))

        # self.points_pos_on_nontarget_limb = {}
        # self.points_nontarget_limb = {}
        # self.total_point_count = 0
        # for limb in self.nontarget_limbs:
        #     length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]
        #     self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
        #     self.points_nontarget_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_nontarget_limb[limb]), visual=True, collision=False, rgba=[0, 1, 1, 1])
        #     self.total_nontarget_point_count += len(self.points_pos_on_nontarget_limb[limb])

        self.update_points_along_body()
    
    def update_points_along_body(self):

        self.points_pos_target_limb_world = {}
        for limb in self.target_limbs:
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])
            points_pos_limb_world = []
            for points_pos_on_target_limb, point in zip(self.points_pos_on_target_limb[limb], self.points_target_limb[limb]):
                point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, points_pos_on_target_limb, [0, 0, 0, 1], physicsClientId=self.id)[0])
                points_pos_limb_world.append(point_pos)
                point.set_base_pos_orient(point_pos, [0, 0, 0, 1])
            self.points_pos_target_limb_world[limb] = points_pos_limb_world


    #!! ONCE EVERYTHING WORKS, GET RID OF THIS    
    """

    def generate_targets(self):

        self.targets_pos_on_foot = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.human.human_lengths['foot']]), radius=self.human.human_radii['foot'], distance_between_points=0.03)
        self.targets_pos_on_shin = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.human.human_lengths['shin']]), radius=self.human.human_radii['shin'], distance_between_points=0.03)

        self.targets_foot = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_foot), visual=True, collision=False, rgba=[0, 1, 0, 1])
        self.targets_shin = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_shin), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.total_target_count = len(self.targets_pos_on_foot) + len(self.targets_pos_on_shin)

        self.update_targets()

    # self.get_quaternion([np.pi/2.0, 0, 0])
    
    def update_targets(self):

        foot_pos, foot_orient = self.human.get_pos_orient(self.human.right_foot)
        foot_pos = foot_pos + [self.human.human_radii['foot']/4, self.human.human_radii['foot']/4, 0]
        print("foot: ", foot_pos, foot_orient)
        foot_orient = self.get_quaternion(self.get_euler(foot_orient) + [-np.pi/2.0, 0, 0])


        # self.create_capsule(radius=self.human.human_radii['foot'], length=0.3, position=foot_pos, orientation=foot_orient)
        print("foot: ", foot_pos, foot_orient)
        self.targets_pos_foot_world = []
        for target_pos_on_foot, target in zip(self.targets_pos_on_foot, self.targets_foot):
            # target_pos = np.array(p.multiplyTransforms(foot_pos, foot_orient, target_pos_on_foot, self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)[0])
            target_pos = np.array(p.multiplyTransforms(foot_pos, foot_orient, target_pos_on_foot, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_foot_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        shin_pos, shin_orient = self.human.get_pos_orient(self.human.right_shin)
        shin_orient = self.get_quaternion(self.get_euler(shin_orient) + [np.pi/30.0, 0, 0])
        # self.create_capsule(radius=self.human.human_radii['shin'], length=0.6, position=shin_pos, orientation=shin_orient)
        print("shin: ", shin_pos, shin_orient)
        self.targets_pos_shin_world = []
        for target_pos_on_shin, target in zip(self.targets_pos_on_shin, self.targets_shin):
            target_pos = np.array(p.multiplyTransforms(shin_pos, shin_orient, target_pos_on_shin, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_shin_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

    
    # ############! NEED TO EDIT FOR CLARITY/TO ADD ALL LIMBS
    def generate_nontargets(self):
        self.target_indices_to_ignore = []
        rgba = [1, 0, 0, 1]
        spacing = 0.03

    
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.human.forearm_length]), radius=self.human.forearm_radius, distance_between_points=spacing)
        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.human.upperarm_length]), radius=self.human.upperarm_radius, distance_between_points=spacing)
        # print(self.targets_pos_on_foot, self.targets_pos_on_shin)

        self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=rgba)
        self.targets_upperarm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=rgba)
        # self.total_target_count = len(self.targets_pos_on_foot) + len(self.targets_pos_on_shin)

        
        self.targets_pos_on_thigh = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.human.thigh_length]), radius=self.human.thigh_radius, distance_between_points=spacing)
        self.targets_thigh = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_thigh), visual=True, collision=False, rgba=rgba)



        self.total_target_count = len(self.targets_pos_on_foot) + len(self.targets_pos_on_shin) + len(self.targets_pos_on_thigh)


        self.update_nontargets()

    
    def update_nontargets(self):
        forearm_pos, forearm_orient = self.human.get_pos_orient(self.human.right_forearm)
        print("forearm: ", forearm_pos, forearm_orient)
        self.targets_pos_forearm_world = []
        for target_pos_on_forearm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_forearm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.human.right_upperarm)
        print("upperarm: ", upperarm_pos, upperarm_orient)
        # print(self.human.get_pos_orient(self.human.right_upperarm))
        self.targets_pos_upperarm_world = []
        for target_pos_on_upperarm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_upperarm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])


        thigh_pos, thigh_orient = self.human.get_pos_orient(self.human.right_thigh)
        print("thigh: ", thigh_pos, thigh_orient)
        thigh_pos = thigh_pos + [self.human.thigh_radius/4, 0, 0]
        thigh_orient = self.get_quaternion(self.get_euler(thigh_orient) + [np.pi/60.0, 0, 0])
        # print(self.human.get_pos_orient(self.human.right_thigh))
        self.targets_pos_thigh_world = []
        for target_pos_on_thigh, target in zip(self.targets_pos_on_thigh, self.targets_thigh):
            target_pos = np.array(p.multiplyTransforms(thigh_pos, thigh_orient, target_pos_on_thigh, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_thigh_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

    """


    
    

