import gym, sys, argparse, multiprocessing, time, os, math
from gym.utils import seeding
from .learn import make_env
import numpy as np
import pickle, pathlib


def sample_action(env):
    return env.action_space.sample()

def gnn_data_collect(env_name, i):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=coop, seed=seeding.create_seed())

    done = False
    #env.render()
    observation = env.reset()
    pid = os.getpid()
    while not done:
        action = sample_action(env)
        # action = np.array([0,0,0,0])
        observation, reward, done, info = env.step(action)
    
    filename = f"c{i}_{env.seed}_pid{pid}"
    with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "observation":observation, 
            "info":info, 
            "action":action}, f)
    output = [i, filename, pid]
    del env
    return output

counter = 0

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Filename: {output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    current_dir = os.getcwd()
    pkl_loc = os.path.join(current_dir,'gnn_test_dc/pickle544')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)


    counter = 0

    # reserve one cpu to keep working while collecting data
    num_processes = multiprocessing.cpu_count() - 1

    # num data points to collect
    trials = 10000

    # trials = 8
    # num_processes = 4

    result_objs = []
    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                result = pool.apply_async(gnn_data_collect, args = (args.env, i), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            # print(len(results))
            # print(results)

    # result_objs = []
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     for i in range(trials):
    #         result = pool.apply_async(gnn_data_collect, args = (args.env, i), callback=counter_callback)
    #         result_objs.append(result)
    
    #     results = [result.get() for result in result_objs]
    #     # print(len(results))
    #     # print(results)
    
    print("Samples Collected")