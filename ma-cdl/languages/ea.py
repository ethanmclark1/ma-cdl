import time
import numpy as np

from math import inf
from scipy import optimize
from languages.utils.cdl import CDL

"""
Evolutionary Algorithm for generating a language
** Does not guarantee optimality **
"""
class EA(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        
    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, scenario):
        degree = 1
        lb, ub = -2, 2
        optim_val, optim_coeffs = inf, None
        start = time.time()
        for num in range(1, 7):
            bounds = [(lb, ub) for _ in range(num*(degree+1))]
            res = optimize.differential_evolution(self._optimizer, bounds, args=(scenario,),
                                                  maxiter=100*num, init='sobol')
            print(f'Cost: {res.fun}')
            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x
        
        end = time.time()
        print(f'Elapsed time: {end - start} seconds')
        optim_coeffs = np.reshape(optim_coeffs, (-1, degree+1))
        return optim_coeffs