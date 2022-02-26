import os, time
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import False_
from numpy.lib.function_base import append
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from .bu_gnn_util import sub_sample_point_clouds, get_body_points_from_obs, get_body_points_reward

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

# since there is no gnn_testing_envs file
from .agents.human import Human
human_controllable_joint_indices = []

#! max_episode_steps=200 CHECK THAT THIS IS STILL OKAY MOVING FORWARD

class BodiesUncoveredGNNEnv(AssistiveEnv):
    def __init__(self):
        super(BodiesUncoveredGNNEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='bedding_manipulation', obs_robot_len=28, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True)
        self.use_mesh = False
        
        self.take_pictures = False
        self.rendering = False
        self.fixed_target = True
        self.target_limb_code = []
        self.target_limb_code = 14
        self.fixed_pose = False
        self.save_pstate = False
        self.pstate_file = None
        self.collect_data = False
        self.blanket_pose_var = False
        self.naive = False

        # self.single_model = True

    def get_naive_action(self, obs):
        
        human_pose = np.reshape(obs, (-1,2))
        feet_midpoint = (human_pose[3] + human_pose[9])/2
        knee_midpoint = (human_pose[4] + human_pose[10])/2
        hip_midpoint = (human_pose[5] + human_pose[11])/2
        btw_upperchest_head_midpoint = (human_pose[12] + human_pose[13])/2

        coeff = np.polyfit([feet_midpoint[1], knee_midpoint[1]], [feet_midpoint[0], knee_midpoint[0]], 1)
        line = np.poly1d(coeff)
        # release_y = list(btw_upperchest_head_midpoint)[1]
        release_y = list(human_pose[13])[1]
        release_x = line(release_y)

        # self.create_sphere(radius=0.01, mass=0.0, pos = list(feet_midpoint)+[0.9], visual=True, collision=True, rgba=[0, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos = list(knee_midpoint)+[0.9], visual=True, collision=True, rgba=[1, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos = list(hip_midpoint)+[0.9], visual=True, collision=True, rgba=[0, 0, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos = list(btw_upperchest_hip_midpoint)+[0.9], visual=True, collision=True, rgba=[0, 0, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos = [release_x, release_y, 0.9], visual=True, collision=True, rgba=[1, 0, 0, 1])

        action = np.array(list(feet_midpoint) + [release_x] + [release_y])
        return action

        
    def step(self, action):
        obs = self._get_obs()
        # return obs, 0, True, {}
        # return obs, -((action[0] - 3) ** 2 + (10 * (action[1] + 2)) ** 2 + (10 * (action[2] + 2)) ** 2 + (10 * (action[3] - 3)) ** 2), 1, {}
        
        if self.rendering:
            print(obs)
            print(action)

        # * scale bounds the 2D grasp and release locations to the area over the mattress (action nums only in range [-1, 1])
        scale = [0.44, 1.05]


        if self.naive:
            # action = self.get_action(obs)
            scale = [1, 1]
        grasp_loc = action[0:2]*scale
        release_loc = action[2:4]*scale

        # * get points on the blanket, initial state of the cloth
        data_i = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist = []
        for i, v in enumerate(data_i[1]):
            v = np.array(v)
            d = np.linalg.norm(v[0:2] - grasp_loc)
            dist.append(d)
        # * if no points on the blanket are within 2.8 cm of the grasp location, exit 
        clipped = False
        if not np.any(np.array(dist) < 0.028):
            clipped = True
            # print("clip")
            # return obs, 0, True, {}
            if self.collect_data:
                return obs, 0, False, {} # for data collect
            else:
                return obs, 0, True, {}

        anchor_idx = np.argpartition(np.array(dist), 4)[:4]
        # for a in anchor_idx:
            # print("anchor loc: ", data[1][a])

        # * update grasp_loc var with the location of the central anchor point on the cloth
        grasp_loc = np.array(data_i[1][anchor_idx[0]][0:2])

        # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
        self.sphere_ee.set_base_pos_orient(data_i[1][anchor_idx[0]], np.array([0,0,0]))
        constraint_ids = []
        constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
        for i in anchor_idx[1:]:
            pos_diff = np.array(data_i[1][i]) - np.array(data_i[1][anchor_idx[0]])
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))

        # * move sphere up by some delta z
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        delta_z = 0.4                           # distance to move up (with respect to the top of the bed)
        bed_height = 0.58                       # height of the bed
        final_z = delta_z + bed_height          # global goal z position
        while current_pos[2] <= final_z:
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]

        # * move sphere to the release location, release the blanket
        travel_dist = release_loc - grasp_loc

        # * determine delta x and y, make sure it is, at max, close to 0.005
        num_steps = np.abs(travel_dist//0.005).max()
        delta_x, delta_y = travel_dist/num_steps

        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        for _ in range(int(num_steps)):
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, 0]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]

        # * continue stepping simulation to allow the cloth to settle before release
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.id)

        # * release the cloth at the release point, sphere is at the same arbitrary z position in the air
        for i in constraint_ids:
            p.removeConstraint(i, physicsClientId=self.id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        # * get points on the blanket, final state of the cloth
        data_f = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        reward = 0
        cloth_initial_subsample = cloth_final_subsample = -1 # none for data collection
        info = {}
        if not self.collect_data:
            reward_distance_btw_grasp_release = -150 if np.linalg.norm(grasp_loc - release_loc) >= 1.5 else 0
            cloth_initial_subsample, cloth_final_subsample = sub_sample_point_clouds(data_i[1], data_f[1])
            cloth_initial_2D = np.delete(np.array(cloth_initial_subsample), 2, axis = 1)
            cloth_final_2D = np.delete(np.array(cloth_final_subsample), 2, axis = 1)
            human_pose = np.reshape(obs, (-1,2))
            all_body_points = get_body_points_from_obs(human_pose, target_limb_code=self.target_limb_code)
            body_point_reward, covered_status = get_body_points_reward(all_body_points, cloth_initial_2D, cloth_final_2D)
            reward = body_point_reward + reward_distance_btw_grasp_release
        else:
            info = {
                "cloth_initial": data_i,
                "cloth_final": data_f,
                "RBG_human": self.human_no_occlusion_RGB,
                "depth_human": self.human_no_occlusion_depth
                # "cloth_initial_subsample": cloth_initial_subsample,
                # "cloth_final_subsample": cloth_final_subsample,
                # "covered_status_sim": self.covered_status
                }
        self.iteration += 1
        done = self.iteration >= 1

        # return 0, 0, 1, {}
        return obs, reward, done, info
    

    def _get_obs(self, agent=None):

        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]

        pose = []
        for limb in self.human.obs_limbs:
            pos, orient = self.human.get_pos_orient(limb)
            # print("pose", limb, pos, orient)
            pos2D = pos[0:2]
            # yaw = p.getEulerFromQuaternion(orient)[-1]
            # pose.append(np.concatenate((pos2D, np.array([yaw])), axis=0))
            pose.append(pos2D)
        pose = np.concatenate(pose, axis=0)

        if self.collect_data:

            output = [None]*28
            all_joint_angles = self.human.get_joint_angles(self.human.all_joint_indices)
            all_pos_orient = [self.human.get_pos_orient(limb) for limb in self.human.all_body_parts]
            output[0], output[1], output[2] = pose, all_joint_angles, all_pos_orient
            return output

            
        # if self.single_model:
        #     one_hot_target_limb = [0]*len(self.human.all_possible_target_limbs)
        #     one_hot_target_limb[self.target_limb_code] = 1
        #     pose = np.concatenate([one_hot_target_limb, pose], axis=0)
        return np.float32(pose)

    def reset(self):

        super(BodiesUncoveredGNNEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed', body_shape=np.zeros((1, 10)))

        # * enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        
        # * Setup human in the air, with legs and arms slightly seperated
        joints_positions = [(self.human.j_left_hip_y, -10), (self.human.j_right_hip_y, 10), (self.human.j_left_shoulder_x, -20), (self.human.j_right_shoulder_x, 20)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])

        if not self.fixed_pose:
            # * Add small variation to the body pose
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            # print(motor_positions)
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.2, 0.2, size=len(motor_indices)))
            # self.increase_pose_variation()
            # * Increase friction of joints so human doesn't fail around exessively as they settle
            # print([p.getDynamicsInfo(self.human.body, joint)[1] for joint in self.human.all_joint_indices])
            self.human.set_whole_body_frictions(spinning_friction=2)

        # * Let the person settle on the bed
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        # * step the simulation a few times so that the human has some initial velocity greater than the at rest threshold
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)
        # * continue stepping the simulation until the human joint velocities are under the threshold
        threshold = 1e-2
        settling = True
        numsteps = 0
        while settling:
            settling = False
            for i in self.human.all_joint_indices:
                if np.any(np.abs(self.human.get_velocity(i)) >= threshold):
                    p.stepSimulation(physicsClientId=self.id)
                    numsteps += 1
                    settling = True
                    break
            if numsteps > 400:
                break
        # print("steps to rest:", numsteps)

        # * Lock the person in place
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
    

        # * select a target limb to uncover (may be fixed or random) 
        # if not self.fixed_target:
        #     self.set_target_limb_code()
        # self.target_limb = self.human.all_possible_target_limbs[self.target_limb_code]

        #! Just to generate points on the body (none considered target)
        # self.target_limb = self.target_limb_code = []

        # self.generate_points_along_body()
        # self.get_point_cloud()
       
        # * spawn blanket
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 1], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)
        if self.blanket_pose_var:
            delta_y = self.np_random.uniform(-0.05, 0.05)
            delta_x = self.np_random.uniform(-0.02, 0.02)
            delta_rad = self.np_random.uniform(-0.0872665, 0.0872665) # 5 degrees

            p.resetBasePositionAndOrientation(self.blanket, [0+delta_x, 0.2+delta_y, 1.5], self.get_quaternion([np.pi/2.0, 0, 0+delta_rad]), physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)


        # self.get_point_cloud()

        # time.sleep(100)


        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # self.non_target_initially_uncovered(data)
        # self.uncover_nontarget_reward(data)

    
        # * Initialize enviornment variables
        # *     if using the sphere manipulator, spawn the sphere and run a modified version of init_env_variables()
        # self.time = time.time()
        if self.robot is None:
            # * spawn sphere manipulator
            position = np.array([-0.3, -0.86, 0.8])
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])
            
            # * initialize env variables
            from gym import spaces
            # * update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 4
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)
            # * Define action/obs lengths
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
        
        if self.save_pstate:
            p.saveBullet(self.pstate_file)
            self.save_pstate = False
        
        
        # * Setup camera for taking images
        # *     Currently saves color images only to specified directory
        if self.take_pictures or self.collect_data:
            self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=398)
            img, depth = self.get_camera_image_depth()
            self.human_no_occlusion_RGB = img
            self.human_no_occlusion_depth = depth
            depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
            depth = (depth * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
            # filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
            # filename = "test_depth.png"
            # cv2.imwrite(os.path.join('/home/mycroft/git/vBMdev/pose_variation_images/lower_var2', filename), img)
            # cv2.imwrite(filename, depth_colormap)

        return self._get_obs()
    
    def generate_points_along_body(self):
        '''
        generate all the target/nontarget posistions necessary to uniformly cover the body parts with points
        if rendering, generates sphere bodies as well
        '''

        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}
        self.total_target_point_count = 0

        self.points_pos_on_nontarget_limb = {}
        self.points_nontarget_limb = {}
        self.total_nontarget_point_count = 0

        #! just for tuning points on torso
        # self.human.all_body_parts = [self.human.waist]

        # * create points on all the body parts
        for limb in self.human.all_body_parts:

            # * get the length and radius of the given body part
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]

            # * create points seperately depending on whether or not the body part is/is a part of the target limb
            # *     generates list of point positions around the body part capsule (sphere if the hands)
            # *     creates all the spheres necessary to uniformly cover the body part (spheres created at some arbitrary position (transformed to correct location in update_points_along_body())
            # *     add to running total of target/nontarget points
            # *     only generate sphere bodies if self.rendering == True
            if limb in self.target_limb:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_target_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.rendering:
                    self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[1, 1, 1, 1])
                self.total_target_point_count += len(self.points_pos_on_target_limb[limb])
            else:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_nontarget_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.rendering:
                    self.points_nontarget_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_nontarget_limb[limb]), visual=True, collision=False, rgba=[0, 0, 1, 1])
                self.total_nontarget_point_count += len(self.points_pos_on_nontarget_limb[limb])

        # * transforms the generated spheres to the correct coordinate space (aligns points to the limbs)
        self.update_points_along_body()
    
    def update_points_along_body(self):
        '''
        transforms the target/nontarget points created in generate_points_along_body() to the correct coordinate space so that they are aligned with their respective body part
        if rendering, transforms the sphere bodies as well
        '''

        # * positions of the points on the target/nontarget limbs in world coordinates
        self.points_pos_target_limb_world = {}
        self.points_pos_nontarget_limb_world = {}

        # * transform all spheres for all the body parts
        for limb in self.human.all_body_parts:

            # * get current position and orientation of the limbs, apply a correction to the pos, orient if necessary
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])
            
            # * transform target/nontarget point positions to the world coordinate system so they align with the body parts
            points_pos_limb_world = []

            if limb in self.target_limb:
                for i in range(len(self.points_pos_on_target_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_target_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    if self.rendering:
                        self.points_target_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_target_limb_world[limb] = points_pos_limb_world
            else:
                for i in range(len(self.points_pos_on_nontarget_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_nontarget_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    if self.rendering:
                        self.points_nontarget_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_nontarget_limb_world[limb] = points_pos_limb_world
