#%%
import gym, sys, argparse, multiprocessing, time, os, math
from gym.utils import seeding
from .learn import make_env
import numpy as np
import pickle, pathlib
from assistive_gym.envs.bu_gnn_util import randomize_target_limbs


# def sample_action(env):
#     return env.action_space.sample()
#%%
def naive_rollout(env_name, i, pkl_loc):
    coop = 'Human' in env_name
    seed = seeding.create_seed()
    target_limb_code = randomize_target_limbs()
    env = make_env(env_name, coop=coop, seed=seed)
    env.naive = True
    env.set_target_limb_code(target_limb_code)

    done = False
    #env.render()
    observation = env.reset()
    cloth_data = env.get_cloth_state()
    while not done:
        action = env.get_naive_action(observation, env.target_limb_code, cloth_data)
        # action = np.array([0,0,0,0])
        observation, reward, done, info = env.step(action)
    
    # return reward
    pid = os.getpid()
    filename = f"c{i}_{seed}_pid{pid}"
    filename = os.path.join(pkl_loc,filename)
    with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "observation":observation, 
            "info":info, 
            "action":action,
            "reward":reward}, f)
    output = [reward, env.target_limb_code]
    del env
    return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"Trial {counter}: Reward = {output[0]:.2f}, TL:{output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='BodiesUncoveredGNN-v1')
    args = parser.parse_args()

    current_dir = os.getcwd()
    pkl_loc = os.path.join(current_dir,'gnn_naive/normal_conditions')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)

    # ! TODO: temporary to prevent messy prints, go back and fix where warning is coming from
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0

    # reserve one cpu to keep working while collecting data
    num_processes = multiprocessing.cpu_count() - 1
    num_processes = 100
    # num_processes = 2

    # num data points to collect
    trials = 500
    # trials = 2

    # trials = 8
    # num_processes = 4
    counter = 0


    # structured this way to prevent unusual errors in pybullet where cloth anchors do not get cleared correctly between trials
    result_objs = []
    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                result = pool.apply_async(naive_rollout, args = (args.env, i, pkl_loc), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            # print(len(results))
            # print(results)
    
    # print(results)
    results_array = np.array(results)
    print("Mean Reward:", np.mean(results_array[:,0]))
    print("Reward Std:", np.std(results_array[:,0]))