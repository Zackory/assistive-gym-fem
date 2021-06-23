import gym, sys, time, argparse
import numpy as np
import pickle
from cmaes import CMA, get_warm_start_mgd

parser = argparse.ArgumentParser(description='CMA-ES sim optimization')
parser.add_argument('--env', default='BeddingManipulationSphere-v1', help='env', required=True)
# parser.add_argument('--cma-dir', default='', help='CMA replay directory', required=True)
args = parser.parse_args()

env = gym.make(args.env)
observation = env.reset()


def source_task(x):
    return 0

def target_task(x):
    global best_params, best_costs, fevals, f
    # Set sim parameters by passing x into the step function
    observation = env.reset()
    done = False
    cost = 0
    while not done:
        # env.render()
        observation, reward, done, info = env.step(x)
        cost = -reward
    return cost

# def target_task(x):
#     return (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2 + (10 * (x[2] + 2)) ** 2 + (10 * (x[3] - 3)) ** 2


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
    f = open(filename,"wb")
    pop_size = 8

    optimizer = CMA(mean=np.zeros(4), sigma=0.05, population_size=pop_size)
    fevals = 0
    iteration = 0
    t0 = time.time()

    # Run WS-CMA-ES
    while True:
        iteration += 1
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            cost = target_task(x)
            t1 = time.time()
            elapsed_time = t1 - t0
            solutions.append((x, cost))
            fevals += 1
            pickle.dump({"iteration": iteration, "fevals": fevals, "elapsed_time": elapsed_time, "reward": -cost, "action": x}, f)
        optimizer.tell(solutions)
        
        mean = np.mean([-cost for x, cost in solutions])
        min = np.min([-cost for x, cost in solutions])
        max = np.max([-cost for x, cost in solutions])
        print(f"Iteration: {iteration}, fevals: {fevals}, elapsed time: {elapsed_time:.2f}, mean reward = {mean:.2f}, min/max reward = {min:.2f}/{max:.2f}")

        if optimizer.should_stop():
            break
    print("Data saved to file:", filename)
    f.close()