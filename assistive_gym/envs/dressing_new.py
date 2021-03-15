import os, time
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DressingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(DressingEnv, self).__init__(robot=robot, human=human, task='dressing', obs_robot_len=(17 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(18 + len(human.controllable_joint_indices)), frame_skip=5, time_step=0.001)
        # self.tt = None

    def step(self, action):
        # if self.tt is not None:
        #     print('Time per iteration:', time.time() - self.tt)
        # self.tt = time.time()
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]

        reward = 0

        obs = self._get_obs()

        if self.gui:
            print('Task success:', self.task_success, 'Average forces on arm:', self.cloth_force_sum)

        info = {'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def _get_obs(self, agent=None):
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        end_effector_pos_real, end_effector_orient_real = self.robot.convert_to_realworld(end_effector_pos, end_effector_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        self.cloth_force_sum = 0
        self.robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        self.total_force_on_human = self.robot_force_on_human + self.cloth_force_sum
        robot_obs = np.concatenate([end_effector_pos_real, end_effector_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.cloth_force_sum]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            end_effector_pos_human, end_effector_orient_human = self.human.convert_to_realworld(end_effector_pos, end_effector_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            human_obs = np.concatenate([end_effector_pos_human, end_effector_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.cloth_force_sum, self.robot_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(DressingEnv, self).reset()
        self.build_assistive_env('wheelchair_left')
        self.cloth_forces = np.zeros((1, 1))
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, np.pi/2.0])

        # Update robot and human motor gains
        # self.robot.motor_gains = self.human.motor_gains = 0.01

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_shoulder_x, -45), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None if self.human.controllable else 1, reactive_gain=0.01)

        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]

        target_ee_pos = np.array([0.45, -0.3, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][0])
        target_ee_orient_shoulder = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][-1])
        offset = np.array([0, 0, 0.1])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos+offset, target_ee_orient_shoulder), (elbow_pos+offset, target_ee_orient), (wrist_pos+offset, target_ee_orient)], arm='left', tools=[], collision_objects=[self.human, self.furniture], right_side=False)
        # if self.robot.mobile:
        #     # Change robot gains since we use numSubSteps=8
        #     self.robot.gains = list(np.array(self.robot.gains) / 8.0)

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        if self.human.controllable or self.human.impairment == 'tremor':
            # Ensure the human arm remains stable while loading the cloth
            self.human.control(self.human.controllable_joint_indices, self.human.get_joint_angles(self.human.controllable_joint_indices), 0.05, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj'), basePosition=[0,0,2], scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections=1, useSelfCollision=0, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj'), basePosition=[0,0,2], mass=0.16, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.01, collisionMargin = 0.006, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)

        self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_660v.obj'), basePosition=[0, 0, 1.5], scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=0, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_2000tri.obj'), basePosition=[0, 0, 1.5], scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=0, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj'), basePosition=[0, 0, 1.5], scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=0, useMassSpring=0, collisionMargin=0.02, useSelfCollision=0, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)

        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        # p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25, physicsClientId=self.id)

        ## self.start_ee_pos, self.start_ee_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        ## self.cloth_attachment = self.create_sphere(radius=0.0001, mass=0, pos=self.start_ee_pos, visual=True, collision=False, rgba=[0, 0, 0, 0], maximal_coordinates=True)
        ## p.createSoftBodyAnchor(self.cloth, 2086, self.cloth_attachment.body, -1, [0, 0, 0], physicsClientId=self.id)

        # data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # for i in range(data[0]):
        #     pos = data[1][i]


        # self.start_ee_pos, self.start_ee_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        # self.cloth_orig_pos = np.array([0.34658437, -0.30296362, 1.20023387])
        # self.cloth_offset = self.start_ee_pos - self.cloth_orig_pos

        # self.cloth_attachment = self.create_sphere(radius=0.0001, mass=0, pos=self.start_ee_pos, visual=True, collision=False, rgba=[0, 0, 0, 0], maximal_coordinates=True)

        # # Load cloth
        # self.cloth = p.loadCloth(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj'), scale=1.4, mass=0.16, position=np.array([0.02, -0.38, 0.84]) + self.cloth_offset/1.4, orientation=self.get_quaternion([0, 0, np.pi]), bodyAnchorId=self.cloth_attachment.body, anchors=[2086, 2087, 2088, 2041], collisionMargin=0.04, rgbaColor=np.array([139./256., 195./256., 74./256., 0.6]), rgbaLineColor=np.array([197./256., 225./256., 165./256., 1]), physicsClientId=self.id)
        # p.clothParams(self.cloth, kLST=0.055, kAST=1.0, kVST=0.5, kDP=0.01, kDG=10, kDF=0.39, kCHR=1.0, kKHR=1.0, kAHR=1.0, piterations=5, physicsClientId=self.id)
        # # Points along the opening of the left arm sleeve
        # self.triangle1_point_indices = [1180, 2819, 30]
        # self.triangle2_point_indices = [1322, 13, 696]

        # p.setGravity(0, 0, -9.81/2, physicsClientId=self.id) # Let the cloth settle more gently
        # if not self.robot.mobile:
        #     self.robot.set_gravity(0, 0, 0)
        # self.human.set_gravity(0, 0, -1)
        # self.cloth_attachment.set_gravity(0, 0, 0)

        # p.setPhysicsEngineParameter(numSubSteps=8, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        # for _ in range(50):
        #     # Force the end effector attachment to stay at the end effector
        #     self.cloth_attachment.set_base_pos_orient(self.start_ee_pos, [0, 0, 0, 1])
        #     p.stepSimulation(physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        # p.setGravity(0, 0, 0, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def update_targets(self):
        # Force the end effector to move forward
        # action = np.array([0, 0.025, 0]) / 10.0
        # ee_pos = self.robot.get_pos_orient(self.robot.left_end_effector)[0] + action
        # ee_pos[-1] = self.start_ee_pos[-1]
        # ik_joint_poses = np.array(p.calculateInverseKinematics(self.robot.body, self.robot.left_end_effector, targetPosition=ee_pos, targetOrientation=self.start_ee_orient, maxNumIterations=100, physicsClientId=self.id))
        # target_joint_positions = ik_joint_poses[self.robot.left_arm_ik_indices]
        # self.robot.set_joint_angles(self.robot.left_arm_joint_indices, target_joint_positions, use_limits=False)

        # Force the end effector attachment to stay at the end effector
        # self.cloth_attachment.set_base_pos_orient(self.robot.get_pos_orient(self.robot.left_end_effector)[0], [0, 0, 0, 1])
        pass

