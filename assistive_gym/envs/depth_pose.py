import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human
from .agents.human_mesh import HumanMesh

human_controllable_joint_indices = human.right_arm_joints + human.left_arm_joints
class DepthPoseEnv(AssistiveEnv):
    def __init__(self):
        # Human Mesh
        # super(DepthPoseEnv, self).__init__(robot=None, human=None, task='human_testing', obs_robot_len=0, obs_human_len=0)
        # Capsulized Human
        super(DepthPoseEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_testing', obs_robot_len=0, obs_human_len=0)

    def step(self, action):
        self.take_step(action, gains=0.05, forces=1.0)
        return [], 0, False, {}

    def _get_obs(self, agent=None):
        return []

    def reset(self):
        super(DepthPoseEnv, self).reset()
        self.build_assistive_env(None, fixed_human_base=False)
        # Human Mesh
        # self.human = None
        # self.robot = None
        
        bed_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 1, 0.2])
        bed_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 1, 0.2])
        # Human Mesh
        # bed_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition = [0, -0.4, 0])
        # Capsulized Human
        bed_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition = [-0.2, -0.1, 0])

        # Human Mesh
        # TODO: setup human mesh in the air and let it settle into a resting pose on the bed instead of setting the base orient directly
        """
        self.human_mesh = HumanMesh()
        h = self.human_mesh
        body_shape = 'female_1.pkl'
        body_shape = self.np_random.randn(1, self.human_mesh.num_body_shape)
        u = self.np_random.uniform
        joint_angles = [(self.human_mesh.j_right_shoulder_z, 60), (self.human_mesh.j_left_shoulder_z, -60)]#, (self.human_mesh.j_right_shoulder_y, u(-45, 45)), (self.human_mesh.j_right_shoulder_z, u(-45, 45))]
        self.human_mesh.init(self.directory, self.id, self.np_random, gender='female', height=1.7, body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[np.pi/2.0, 0, 0])
        self.human_mesh.set_base_pos_orient([0, 0, 0.35], [0, 0, 0, 1])

        p.setGravity(0, 0, -1, physicsClientId=self.id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        self.human_mesh.set_gravity(0, 0, -1)
        """

        # Capsulized Human
        #"""
        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        self.human.set_gravity(0, 0, -1)
        human_height, human_base_height = self.human.get_heights()
        print('Human height:', human_height, 'm')
        #"""

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=5, mass=0.1, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.001, frictionCoeff=0.25, useFaceContact=1, physicsClientId=self.id)
        # TODO: replace cloth with irregular_plane.obj with fewer vertices because the current .obj file is too big and crashes pybullt
        self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'), scale=3, mass=0.1, useNeoHookean=0, useBendingSprings=0, useMassSpring=0, springElasticStiffness=0, springDampingStiffness=0, springDampingAllDirections=0, springBendingStiffness=0, useSelfCollision=0, collisionMargin=0, frictionCoeff=0, useFaceContact=0, physicsClientId=self.id)
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        
        # TODO: make sure depth camera parameters are what henry needs
        AssistiveEnv.setup_camera_rpy(self, camera_target=[-0.2, 0, 0.75], distance=2.101, rpy=[90, -60, 0], fov=60, camera_width=128, camera_height=54)
        print("camera image depth: ", AssistiveEnv.get_camera_image_depth(self))

        p.resetDebugVisualizerCamera(cameraDistance=2.101, cameraYaw=90, cameraPitch=-60, cameraTargetPosition=[0, 0, 1.21], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # self.init_env_variables()
        return self._get_obs()