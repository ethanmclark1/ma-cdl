import time
import numpy as np

from math import inf
from scipy import optimize
from languages.utils.cdl import CDL

"""Evolutionary Algorithm"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
                
    def optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super().optimizer(regions, problem_instance)
        return scenario_cost
        
    # Minimize cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, problem_instance):
        start_time = time.time()
        lb, ub = -1, 1
        optim_val, optim_coeffs = inf, None
        for num in range(self.min_lines, self.max_lines + 1):
            bounds = [(lb, ub) for _ in range(3*num)]
            res = optimize.differential_evolution(self.optimizer, bounds, args=(problem_instance,),
                                                     tol=0.04, maxiter=175*num, init='sobol')

            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x
        
        optim_coeffs = np.reshape(optim_coeffs, (-1, 3))
        return optim_coeffs