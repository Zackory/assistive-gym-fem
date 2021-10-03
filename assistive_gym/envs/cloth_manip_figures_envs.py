from .cloth_manip_figures import ClothManipEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.body_joints + human.right_arm_joints
class ClothManipStretchEnv(ClothManipEnv):
    def __init__(self):
        super(ClothManipStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

