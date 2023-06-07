import time
import numpy as np

from math import inf
from scipy import optimize
from languages.utils.cdl import CDL

"""Evolutionary Algorithm"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
                
    def _optimizer(self, coeffs, instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super()._optimizer(regions, instance)
        return scenario_cost
        
    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, instance):
        lb, ub = -1, 1
        optim_val, optim_coeffs = inf, None
        start = time.time()
        for num in range(self.min_lines, self.max_lines):
            bounds = [(lb, ub) for _ in range(3*num)]
            res = optimize.differential_evolution(self._optimizer, bounds, args=(instance,),
                                                  maxiter=100*num, init='sobol')
            print(f'Cost: {res.fun}')
            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x
        
        end = time.time()
        print(f'Elapsed time: {end - start} seconds')
        optim_coeffs = np.reshape(optim_coeffs, (-1, 3))
        return optim_coeffs