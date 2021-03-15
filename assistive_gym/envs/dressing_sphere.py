import os, time
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DressingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(DressingEnv, self).__init__(robot=robot, human=human, task='dressing', obs_robot_len=(17 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(18 + len(human.controllable_joint_indices)), frame_skip=5, time_step=0.001)

    def step(self, action):
        # for j in self.robot.left_arm_joint_indices:
        #     p.applyExternalForce(self.robot.body, j, forceObj=[0, 0, -self.gravity*self.robot.get_mass(j)], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.id)
        #     # p.applyExternalForce(self.robot.body, j, forceObj=[0, 0, -self.gravity*self.robot.get_mass(j)], posObj=self.robot.get_pos_orient(j, center_of_mass=True)[0], flags=p.WORLD_FRAME, physicsClientId=self.id)

        self.take_step(np.zeros(7))
        # self.take_step(np.array([0, -1, 0, 0, 0, 0, 0]))

        # Get cloth data
        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        # print(mesh_points)
        # print(forces)
        i = 0
        for cp, f in zip(contact_positions, forces):
            if not np.array_equal(f, np.zeros(3)):
                self.points[i].set_base_pos_orient(cp, [0, 0, 0, 1])
                print(f)
                i += 1
        for j in range(i, len(self.points)):
            self.points[j].set_base_pos_orient([100, 100+j, 100], [0, 0, 0, 1])

        # print(self.robot.get_force_torque_sensor(self.robot.left_end_effector-1)[:3])

        return np.zeros(1), 0, False, {}

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(DressingEnv, self).reset()

        # self.robot = None
        self.human = None
        self.build_assistive_env(None)

        ## self.robot.motor_forces = 100.0
        self.robot.set_base_pos_orient([1, -0.5, 0.96], [0, 0, np.pi])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_2000tri.obj'), basePosition=[0, 0, 2], scale=1.0, mass=1.0, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=1, springBendingStiffness=0.1, useSelfCollision=0, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)

        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_660v.obj'), basePosition=[0, 0, 1.5], scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=1, collisionMargin = 0.001, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)
        self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_660v.obj'), scale=1.4, mass=0.16, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, springElasticStiffness=100, springDampingStiffness=0.1, springDampingAllDirections=0, springBendingStiffness=0.1, useSelfCollision=1, collisionMargin = 0.001, frictionCoeff=.5, useFaceContact=1, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.5], flags=0)

        vert_pos = []
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        for i in range(data[0]):
            vert_pos.append(data[1][i])
        vert_pos = np.array(vert_pos)
        indices = vert_pos[:, 0] < 0.05
        indices = np.logical_and(indices, vert_pos[:, 0] > -0.05)
        indices = np.logical_and(indices, vert_pos[:, 1] < 0.05)
        indices = np.logical_and(indices, vert_pos[:, 1] > -0.05)
        indices = np.logical_and(indices, vert_pos[:, 2] < 0.05)
        indices = np.logical_and(indices, vert_pos[:, 2] > -0.05)
        print('Indices:', [i for i, idx in enumerate(indices) if idx])
        print(len(vert_pos[indices]), vert_pos[indices])
        vertex_index = [i for i, idx in enumerate(indices) if idx][0]
        vertex_position = vert_pos[indices][0]

        offset = np.array([0, 0, 1.5])
        # offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position

        p.resetBasePositionAndOrientation(self.cloth, offset, [0, 0, 0, 1], physicsClientId=self.id)

        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        new_vertex_position = data[1][vertex_index]

        # self.cloth_attachment = self.create_sphere(radius=0.02, mass=0, pos=new_vertex_position, visual=True, collision=False, rgba=[0, 0, 0, 1], maximal_coordinates=False)
        # p.createSoftBodyAnchor(self.cloth, vertex_index, self.cloth_attachment.body, -1, [0, 0, 0], physicsClientId=self.id)
        # p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)

        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25, physicsClientId=self.id)

        self.sphere = self.create_sphere(radius=0.02, mass=0, pos=[-0.2, 0, 1.4], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False)

        self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)

        batch_positions = []
        for i in range(100):
            batch_positions.append(np.array([100, 100+i, 100]))
        self.points = self.create_spheres(radius=0.01, mass=0, batch_positions=batch_positions, visual=True, collision=False, rgba=[1, 1, 1, 1])

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        # p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        # p.setGravity(0, 0, 0, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

