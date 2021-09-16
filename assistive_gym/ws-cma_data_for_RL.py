import os, gym, sys, time, argparse
import numpy as np
import pickle
from cmaes import CMA, get_warm_start_mgd

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

parser = argparse.ArgumentParser(description='CMA-ES sim optimization')
parser.add_argument('--env', default='BeddingManipulationSphere-v1', help='env', required=True)
# parser.add_argument('--cma-dir', default='', help='CMA replay directory', required=True)
args = parser.parse_args()

env = gym.make(args.env)
observation = env.reset()

#!!!!!!!!!!!!!!!!!!!!!!!!NEEDS TO BE TESTED!! <- must have fixed pose and target limb
'''
see:
https://github.com/ray-project/ray/blob/master/rllib/examples/saving_experiences.py
reference used to set up saving batch experiences
'''


# similar task whose output is used to warm start cma-es, empty at the moment
def source_task(x):
    return 0

# cost function
def target_task(action):
    observation = env.reset()
    done = False
    while not done:
        # env.render()
        observation, reward, done, info = env.step(action)
    return observation, reward, done, info


# test function - not fully tested yet
# def target_task(action):
#     reward = (action[0] - 3) ** 2 + (10 * (action[1] + 2)) ** 2 + (10 * (action[2] + 2)) ** 2 + (10 * (action[3] - 3)) ** 2
#     return 0, reward, 1, {}


if __name__ == "__main__":
    ''' 
    # SAMPLE CODE FOR WARM START
    # Generate solutions from a source task
    source_solutions = []
    for _ in range(1000):
        x = np.random.random(2)
        value = source_task(x[0], x[1])
        source_solutions.append((x, value))

    # Estimate a promising distribution of the source task
    ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(
        source_solutions, gamma=0.1, alpha=0.1
    )
    '''
    filename = "ws-cmaes" + time.strftime("%y%m%d-%H%M") + '.pkl'
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(os.path.join(ray._private.utils.get_user_temp_dir(), filename))

    #! May or may not need to do this
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    
    # set up optimizer
    pop_size = 8
    optimizer = CMA(mean=np.zeros(4), sigma=0.05, population_size=pop_size)


    fevals = 0      # running total of sim steps (over all iterations)
    iteration = 0
    t0 = time.time()

    # Run WS-CMA-ES
    # continue running until optimizer meets convergence/termination criteria
    while True:
        iteration += 1
        solutions = []
        prev_action = np.zeros(4)
        prev_reward = 0

        # collect actions and costs for pop_size number of function evaluations (simulations)
        for _ in range(optimizer.population_size):
            action = optimizer.ask()
            
            observation = env.reset()
            
            new_observation, reward, done, info = target_task(action)
            cost = -reward

            t1 = time.time()
            elapsed_time = t1 - t0
            solutions.append((action, cost))
            fevals += 1
            
            # 
            # In this scenario, current and new observation are the same
            batch_builder.add_values(
                t = 1,              # number of timesteps (only 1 because only taking a single step)
                eps_id = fevals,
                agent_index = 0, #! look into
                obs=prep.transform(observation), #! look into
                actions=action,
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_observation))
            
            writer.write(batch_builder.build_and_reset())

        # tell the optimizer what the actions and costs for the popsize number of function evaluations (simulations) were
        optimizer.tell(solutions)
        
        mean = np.mean([-cost for action, cost in solutions])
        min = np.min([-cost for action, cost in solutions])
        max = np.max([-cost for action, cost in solutions])
        print(f"Iteration: {iteration}, fevals: {fevals}, elapsed time: {elapsed_time:.2f}, mean reward = {mean:.2f}, min/max reward = {min:.2f}/{max:.2f}")

        if optimizer.should_stop():
            break