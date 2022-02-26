import gym, sys, argparse, multiprocessing, time, os, math
from gym.utils import seeding
from .learn import make_env
import numpy as np
import pickle, pathlib


# def sample_action(env):
#     return env.action_space.sample()

def naive_rollout(env_name, i):
    coop = 'Human' in env_name
    seed = seeding.create_seed()
    env = make_env(env_name, coop=coop, seed=seed)
    env.naive = True

    done = False
    #env.render()
    observation = env.reset()
    while not done:
        action = env.get_naive_action(observation)
        # action = np.array([0,0,0,0])
        observation, reward, done, info = env.step(action)
    
    return reward
    # filename = f"c{i}_{seed}_pid{pid}"
    # with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
    #     pickle.dump({
    #         "observation":observation, 
    #         "info":info, 
    #         "action":action}, f)
    # output = [i, filename, pid]
    # del env
    # return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"Trial {counter}: Reward = {output:.2f}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='BodiesUncoveredGNN-v1')
    args = parser.parse_args()

    current_dir = os.getcwd()
    pkl_loc = os.path.join(current_dir,'gnn_new_data/akhil')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)

    # ! TODO: temporary to prevent messy prints, go back and fix where warning is coming from
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0

    # reserve one cpu to keep working while collecting data
    num_processes = multiprocessing.cpu_count() - 1

    # num data points to collect
    trials = 100

    # trials = 8
    # num_processes = 4
    counter = 0


    # structured this way to prevent unusual errors in pybullet where cloth anchors do not get cleared correctly between trials
    result_objs = []
    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                result = pool.apply_async(naive_rollout, args = (args.env, i), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            # print(len(results))
            # print(results)
    
    print(results)
    print("Mean Reward:", np.mean(results))
    print("Reward Std:", np.std(results))