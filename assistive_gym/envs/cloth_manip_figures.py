import os, time
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from .agents.human_mesh import HumanMesh

class ClothManipEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ClothManipEnv, self).__init__(robot=robot, human=None, task='bed_bathing', obs_robot_len=(1 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(1 + len(human.controllable_joint_indices)), time_step=0.02, deformable=True)

    def step(self, action):
        self.take_step(np.zeros_like(action))
        obs = self._get_obs()
        reward = 0
        info = {}
        done = self.iteration >= 200
        return obs, reward, done, info

    def _get_obs(self, agent=None):
        return np.array([0])

    def reset(self):
        super(ClothManipEnv, self).reset()

        bedding = False

        self.human = HumanMesh()
        if bedding:
            self.build_assistive_env('hospital_bed', fixed_human_base=False)
            self.furniture.set_on_ground()
            self.furniture.set_whole_body_frictions(1, 1, 1)
            # Enable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

            joints_positions = [(self.human.j_right_shoulder_z, 70), (self.human.j_left_shoulder_z, -70)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions)
            self.human.set_base_pos_orient([0, 0.2, 0.8], [-np.pi/2.0, 0, 0])
            # Drop human on bed
            p.setGravity(0, 0, -1, physicsClientId=self.id)
            self.human.set_mass(-1, mass=1)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.id)
                time.sleep(0.02)
            self.human.set_mass(-1, mass=0)

            pos = np.array(self.robot.toc_base_pos_offset[self.task]) + np.array([0.35, 0.3, 0])
            orient = np.array(self.robot.toc_ee_orient_rpy[self.task])
            self.robot.set_base_pos_orient(pos, orient)
            self.robot.set_joint_angles([3], [1.0])
            self.robot.set_joint_angles([5, 6, 7, 8], [0.05]*4)

            # Spawn blanket
            self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
            # change alpha value so that it is a little more translucent, easier to see the relationship the human
            # p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.5], flags=0, physicsClientId=self.id)
            p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
            p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
            p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=4, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.blanket, [0, -0.5, 1], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
            vertex_index = 516
            vertex_index = 73

            # Move cloth grasping vertex into robot end effector
            # p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
            data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            vertex_position = np.array(data[1][vertex_index])
            offset = self.robot.get_pos_orient(self.robot.right_end_effector)[0] - vertex_position + np.array([0, -0.5, 1])
            p.resetBasePositionAndOrientation(self.blanket, offset, self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
            data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            new_vertex_position = np.array(data[1][vertex_index])

            # NOTE: Create anchors between cloth and robot end effector
            p.createSoftBodyAnchor(self.blanket, vertex_index, self.robot.body, self.robot.right_end_effector, [0, 0, 0], physicsClientId=self.id)
            for i, v_pos in enumerate(data[1]):
                if np.linalg.norm(new_vertex_position - np.array(v_pos)) < 0.05:
                    pos_diff = v_pos - new_vertex_position
                    p.createSoftBodyAnchor(self.blanket, i, self.robot.body, self.robot.right_end_effector, pos_diff, physicsClientId=self.id)
            # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            # vertex_position = np.array(data[1][vertex_index])
            # offset = self.robot.get_pos_orient(self.robot.right_end_effector)[0] - vertex_position
            # p.createSoftBodyAnchor(self.blanket, vertex_index, self.robot.body, self.robot.right_end_effector, [0, 0, 0], physicsClientId=self.id)
            # for i in anchor_vertices:
            #     pos_diff = np.array(data[1][i]) - new_vertex_position
            #     p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

            # Drop the blanket on the person, allow to settle
            p.setGravity(0, 0, -9.81, physicsClientId=self.id)
            for _ in range(20):
                p.stepSimulation(physicsClientId=self.id)

        else:
            self.build_assistive_env('wheelchair2')
            self.furniture.set_on_ground()
            p.changeVisualShape(self.furniture.body, -1, rgbaColor=[0.3, 0.3, 0.3, 1], flags=0, physicsClientId=self.id)
            # Enable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

            # joints_positions = [(self.human.j_right_shoulder_z, 10), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -60), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 70), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 70)]
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -15), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 60), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 60)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])
            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])

            # pos = np.array(self.robot.toc_base_pos_offset[self.task]) + np.array([0.15, 0.1, 0])
            pos = np.array(self.robot.toc_base_pos_offset[self.task]) + np.array([2.03, 0.25, 0])
            orient = np.array(self.robot.toc_ee_orient_rpy[self.task]) + np.array([0, 0, np.pi])
            self.robot.set_base_pos_orient(pos, orient)
            self.robot.set_joint_angles([3], [1.0])
            self.robot.set_joint_angles([5, 6, 7, 8], [0.05]*4)

            self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'gown_696v.obj'), scale=1.0, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
            p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.75], flags=0, physicsClientId=self.id)
            # p.changeVisualShape(self.cloth, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
            p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
            p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)
            vertex_index = 680
            anchor_vertices = [307, 300, 603, 43, 641, 571]
            # vertex_index = 393

            # Move cloth grasping vertex into robot end effector
            p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)
            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            vertex_position = np.array(data[1][vertex_index])
            offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
            p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)
            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            new_vertex_position = np.array(data[1][vertex_index])

            # NOTE: Create anchors between cloth and robot end effector
            p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)
            for i in anchor_vertices:
                pos_diff = np.array(data[1][i]) - new_vertex_position
                p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)


        # self.furniture.set_joint_angles([1], [np.pi/4])

        # Open gripper to hold the tool
        # self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)

        self.init_env_variables()
        return self._get_obs()

