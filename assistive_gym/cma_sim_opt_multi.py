import gym, sys, argparse, multiprocessing, time
import numpy as np
import cma
from cma.optimization_tools import EvalParallel2
import pickle
# import assistive_gym


def cost_function(action):
    t0 = time.time()
    # Set sim parameters by passing x into the step function
    observation = env.reset()
    done = False
    while not done:
        # env.render()
        observation, reward, done, info = env.step(action)
        t1 = time.time()
        cost = -reward
        elapsed_time = t1 - t0

    return [cost, observation, elapsed_time]


# def cost_function(action):
#     t0 = time.time()

#     done = False
#     cost = 0
#     while not done:
#         t1 = time.time()
#         cost = (action[0] - 3) ** 2 + (10 * (action[1] + 2)) ** 2 + (10 * (action[2] + 2)) ** 2 + (10 * (action[3] - 3)) ** 2
#         observation = 0
#         elapsed_time = t1 - t0
#         done = True

#     return [cost, observation, elapsed_time]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMA-ES sim optimization')
    parser.add_argument('--env', default='BeddingManipulationSphere-v1', help='env', required=True)
    args = parser.parse_args()

    env = gym.make(args.env)

    fevals = 0
    iterations = 0

    filename = "cmaes_multi" + time.strftime("%y%m%d-%H%M") + '.pkl'
    f = open(filename,"wb")

    num_proc = multiprocessing.cpu_count()
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': num_proc}) # , 'tolfun': 10, 'maxfevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)

    x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.1
    reward_threshold = 95
    t0 = time.time()

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    with EvalParallel2(cost_function, number_of_processes=num_proc) as eval_all:
        while not es.stop():
            iterations += 1
            fevals += num_proc
            
            action = es.ask()
            output = eval_all(action)
            t1 = time.time()
            output = [list(x) for x in zip(*output)]
            costs = output[0]
            observations = output[1]
            elapsed_time = output[2]
            es.tell(action, costs)
            rewards = [-c for c in costs]

            mean = np.mean(rewards)
            min = np.min(rewards)
            max = np.max(rewards)
            total_elapsed_time = t1-t0


            print(f"Iteration: {iterations}, fevals: {fevals}, elapsed time: {total_elapsed_time:.2f}, mean reward = {mean:.2f}, min/max reward = {min:.2f}/{max:.2f}")
            pickle.dump({"iteration": iterations, "fevals": fevals, "total_elapsed_time":total_elapsed_time, "rewards": rewards, "observations":observations, "elapsed_time": elapsed_time}, f)

            if np.any(np.array(costs) <= -reward_threshold): break

    es.result_pretty()
    print("Data saved to file:", filename)
    f.close()
