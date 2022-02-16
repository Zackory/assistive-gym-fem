import argparse, multiprocessing
from gym.utils import seeding
from .learn import make_env

# * samples a random action from the envionment's action space
def sample_action(env):
    return env.action_space.sample()


# * create, reset, and step the assisitve enviornment, return the reward achieved
def run_env(env_name, i):

    # * build the assistive enviornment
    coop = 'Human' in env_name
    env = make_env(env_name, coop=coop, seed=seeding.create_seed())

    done = False
    #env.render()   # suppress rendering while processing in parallel

    # * capture observation of the assitive enviornment
    observation = env.reset()

    # * step enviornment until done (max iterations reached)
    while not done:
        action = sample_action(env)
        observation, reward, done, info = env.step(action)
    
    result = reward
    return result

# * callback prints running count as rollouts are completed
def counter_callback(result):
    global counter
    counter += 1
    print(f"Trial {counter} - Reward: {result}")


if __name__ == "__main__":
    # * capture enviornment to run and number of times to run the enviornment from CMD args
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    parser.add_argument('--rollouts', type=int, default='ScratchItchJaco-v1',
                        help='Number of times to run the assistive enviornment')
    args = parser.parse_args()

    # * set the number of parallel processes to run equal to the number of cpus on the machine
    num_processes = multiprocessing.cpu_count()

    # * run 1 rollout per cpu until all rollouts are complete, result variable keeps a list of all output from run_env()
    counter = 0
    result_objs = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i in range(args.rollouts):
            result = pool.apply_async(run_env, args = (args.env, i), callback=counter_callback)
            result_objs.append(result)

        results = [result.get() for result in result_objs]
    
    # print(results)