import os, time
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DressingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(DressingEnv, self).__init__(robot=robot, human=human, task='dressing', obs_robot_len=(17 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(18 + len(human.controllable_joint_indices)), frame_skip=1, time_step=0.01)

    def step(self, action):
        # self.take_step(np.zeros(7))
        # self.take_step(np.array([0, -1, 0, 0, 0, 0, 0]))
        # self.take_step(action)

        pos, orient = self.cloth_attachment.get_base_pos_orient()
        self.cloth_attachment.set_base_pos_orient(pos + np.array([0, 0.002, 0]), orient)

        # pos, orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        # pos[0] = self.start_pos[0]
        # pos[-1] = self.start_pos[-1]
        # target_joint_angles = self.robot.ik(self.robot.left_end_effector, pos + np.array([0, 0.002, 0]), orient, self.robot.left_arm_ik_indices, max_iterations=200, use_current_as_rest=True)
        # self.robot.set_joint_angles(self.robot.controllable_joint_indices, target_joint_angles, use_limits=False)
        # # current_joint_angles = self.robot.get_joint_angles(self.robot.left_arm_joint_indices)
        # # action = (target_joint_angles - current_joint_angles) * 10
        # # self.take_step(action)
        self.take_step(np.zeros(7))
        print('Time:', time.time() - self.time)
        self.time = time.time()


        # Get cloth data
        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        # print(mesh_points)
        # print(forces)
        total_force = 0
        i = 0
        for cp, f in zip(contact_positions, forces):
            if i >= len(self.points):
                break
            if not np.array_equal(f, np.zeros(3)):
                self.points[i].set_base_pos_orient(cp, [0, 0, 0, 1])
                total_force += np.linalg.norm(f)
                i += 1
        print('Force:', total_force)
        for j in range(i, len(self.points)):
            self.points[j].set_base_pos_orient([100, 100+j, 100], [0, 0, 0, 1])

        # print(self.robot.get_force_torque_sensor(self.robot.left_end_effector-1)[:3])

        return np.zeros(1), 0, False, {}

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(DressingEnv, self).reset()
        self.build_assistive_env('wheelchair_left')
        self.cloth_forces = np.zeros((1, 1))
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, np.pi/2.0])

        # Update robot and human motor gains
        # self.robot.motor_forces = self.human.motor_forces = 5.0
        self.robot.motor_gains = self.human.motor_gains = 0.01

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_shoulder_x, -80), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        # self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None if self.human.controllable else 1, reactive_gain=0.01)
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]

        target_ee_pos = np.array([0.41, -0.3, 1.02])
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][0])
        target_ee_orient_shoulder = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][-1])
        offset = np.array([0, 0, 0.1])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos+offset, target_ee_orient_shoulder), (elbow_pos+offset, target_ee_orient), (wrist_pos+offset, target_ee_orient)], arm='left', tools=[], collision_objects=[self.human, self.furniture], right_side=False)
        self.start_pos, self.start_orient = self.robot.get_pos_orient(self.robot.left_end_effector)

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        if self.human.controllable or self.human.impairment == 'tremor':
            # Ensure the human arm remains stable while loading the cloth
            self.human.control(self.human.controllable_joint_indices, self.human.get_joint_angles(self.human.controllable_joint_indices), 0.05, 1)




        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_2000tri.obj'), scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=10, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_660v.obj'), scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.5, springDampingAllDirections=0, springBendingStiffness=0.01, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_660v.obj'), scale=1.4, mass=0.16, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.5, springDampingAllDirections=0, springBendingStiffness=0.01, useNeoHookean=1, NeoHookeanMu=180, NeoHookeanLambda=600, NeoHookeanDamping=0.01, repulsionStiffness=800, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_2000tri.obj'), scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=1.0, springDampingAllDirections=0, springBendingStiffness=0.01, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=1.0, mass=0.1, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=1, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=0.75, mass=0.1, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.25, useFaceContact=1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=0.75, mass=0.1, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=10, springDampingStiffness=1, springDampingAllDirections=0, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.25, useFaceContact=1, physicsClientId=self.id)
        self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=0.75, mass=0.1, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.25, useFaceContact=1, physicsClientId=self.id)

        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.5], flags=0)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=8, numSolverIterations=1, physicsClientId=self.id)
        # p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25, physicsClientId=self.id)
        # p.clothParams(self.cloth, kLST=0.055, kAST=1.0, kVST=0.5, kDP=0.01, kDG=10, kDF=0.39, kCHR=1.0, kKHR=1.0, kAHR=1.0, piterations=5, physicsClientId=self.id)
        # p.clothParams(self.cloth, kDP=0.01, kDG=10, physicsClientId=self.id)
        # p.clothParams(self.cloth, kDG=1, physicsClientId=self.id)

        # vertex_index = 80 # 866, 649, 1141, 1070
        # Triangle1: 1091, 721, 369
        # Triangle2: 521, 983, 44
        # vertex_index = 1 # 525, 581
        vertex_index = 576 # 483, 484, 575, 577, 476, 560, 467, 468

        # Move cloth grasping vertex into robot end effector
        p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        new_vertex_position = np.array(data[1][vertex_index])

        # p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)
        # # for i in [866, 649, 1141, 1070]:
        # for i in [525, 581]:
        #     pos_diff = np.array(data[1][i]) - new_vertex_position
        #     p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

        self.cloth_attachment = self.create_sphere(radius=0.02, mass=0, pos=new_vertex_position, visual=True, collision=False, rgba=[0, 0, 0, 1], maximal_coordinates=False)
        p.createSoftBodyAnchor(self.cloth, vertex_index, self.cloth_attachment.body, -1, [0, 0, 0], physicsClientId=self.id)
        # for i in [866, 649, 1141, 1070]:
        # for i in [525, 581]:
        for i in [483, 484, 575, 577, 476, 560, 467, 468]:
            pos_diff = np.array(data[1][i]) - new_vertex_position
            p.createSoftBodyAnchor(self.cloth, i, self.cloth_attachment.body, -1, pos_diff, physicsClientId=self.id)

        self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)

        # Disable collisions between robot and cloth
        for i in range(-1, p.getNumJoints(self.robot.body, physicsClientId=self.id)):
            p.setCollisionFilterPair(self.robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)
        p.setCollisionFilterPair(self.furniture.body, self.cloth, -1, -1, 0, physicsClientId=self.id)

        batch_positions = []
        for i in range(100):
            batch_positions.append(np.array([100, 100+i, 100]))
        self.points = self.create_spheres(radius=0.01, mass=0, batch_positions=batch_positions, visual=True, collision=False, rgba=[1, 1, 1, 1])

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        # p.setGravity(0, 0, 0, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        # p.setGravity(0, 0, -1, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)
        # p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        print('Settled')

        self.time = time.time()
        self.init_env_variables()
        return self._get_obs()

