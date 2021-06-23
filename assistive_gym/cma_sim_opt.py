import gym, sys, argparse, cma
import numpy as np
import pickle
# import assistive_gym

parser = argparse.ArgumentParser(description='CMA-ES sim optimization')
parser.add_argument('--env', default='BeddingManipulationSphere-v1', help='env', required=True)
# parser.add_argument('--cma-dir', default='', help='CMA replay directory', required=True)
args = parser.parse_args()

env = gym.make(args.env)
observation = env.reset()

best_params = []
best_costs = []
fevals = 0
f = open("all_cmaes_data","wb")

def cost_function(x):
    global best_params, best_costs, fevals, f
    # Set sim parameters by passing x into the step function
    observation = env.reset()
    done = False
    cost = 0
    while not done:
        # env.render()
        observation, reward, done, info = env.step(x)
        cost = -reward

    # Keep track of the best 5 param sets
    if len(best_costs) < 5 or cost < best_costs[-1]:
        # Find location to insert new low cost param set
        index = 0
        for i in range(len(best_costs)):
            if cost < best_costs[i]:
                index = i
                break
        best_params.insert(index, x)
        best_costs.insert(index, cost)
        if len(best_costs) > 5:
            best_params.pop(-1)
            best_costs.pop(-1)
    if fevals % 21 == 0:
        # if fevals % 50 == 0:
        # Print out best param sets so far
        print('Best Costs:', best_costs)
        print('Best Params:', [[('%.5f' % xx) for xx in x] for x in best_params])
    fevals += 1

    # all_data.append([fevals, x, cost])

    pickle.dump([fevals, x, cost], f)

    return cost

opts = cma.CMAOptions({'verb_disp': 1, 'popsize': 8}) # , 'tolfun': 10, 'maxfevals': 500
# opts = cma.CMAOptions({'verb_disp': 1, 'popsize': 25}) # , 'tolfun': 10, 'maxfevals': 500
# bound = np.array([0.3, 1.0, 5, 1, 2])
# opts.set('bounds', [[0.0]*5, bound])
# bound = np.array([0.3]*7 + [1.0]*7 + [5]*7 + [1]*7 + [2]*7)
# bound = np.array([0.3]*7 + [1.0]*7 + [5]*7 + [1]*7 + [5]*7)
# opts.set('bounds', [[0.0]*35, bound])
bound = np.array([1]*4)
opts.set('bounds', [[-1]*4, bound])
opts.set('CMA_stds', bound)
x0 = np.random.uniform(-1,1,4)
# sigma0 = 0.3 # bound/4.0
sigma0 = 0.05 # bound/4.0
xopt, es = cma.fmin2(cost_function, x0, sigma0, opts)
es.result_pretty()


print('Best Costs:', best_costs)
print('Best Params:', [[('%.5f' % xx) for xx in x] for x in best_params])

f.close()