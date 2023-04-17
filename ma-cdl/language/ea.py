import time
import numpy as np

from math import inf
from scipy import optimize
from shapely.geometry import Point
from statistics import mean, variance

from language.utils.cdl import CDL

"""
Evolutionary Algorithm for generating a language
** Does not guarantee optimality **
"""
class EA(CDL):
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        super().__init__(agent_radius, obs_radius, num_obstacles)
        self.configs_to_consider = 30
        
    """
    Calculate cost of a configuration (i.e. start, goal, and obstacles)
    with respect to the regions and positions under some positional constraints: 
        1. Unsafe area caused by obstacles
        2. Unsafe plan caused by non-existent path from start to goal while avoiding unsafe area
    """
    def _config_cost(self, start, goal, obstacles, regions):
        obstacles_idx = set(idx for idx, region in enumerate(regions)
                            for obs in obstacles if region.contains(Point(obs)))
        nonnavigable = sum(regions[idx].area for idx in obstacles_idx)
        
        path = self.rrt_star.plan(start, goal, obstacles)
        if path is None: return None

        unsafe = 0
        num_path_checks = 15
        path_length = len(path)
        significand = path_length // num_path_checks
        
        for idx in range(num_path_checks):
            path_point = path[idx * significand]
            for region_idx, region in enumerate(regions):
                if region.contains(Point(path_point)) and region_idx in obstacles_idx:
                    unsafe += 1
                    break

        return nonnavigable, unsafe
    
    """ 
    Calculate cost of a given problem (i.e. all configurations) 
    with respect to the regions and the given positional constraints: 
        1. Unsafe plans
        2. Language efficiency
        3. Mean of nonnavigable area
        4. Variance of nonnavigable area
    """
    def _optimizer(self, coeffs, scenario):
        lines = self._get_lines_from_coeffs(coeffs)
        regions = self._create_regions(lines)
        if len(regions) == 0: return inf
        
        i = 0
        nonnavigable, unsafe = [], []
        while i < self.configs_to_consider:
            start, goal, obstacles = self._generate_points(scenario)
            config_cost = self._config_cost(start, goal, obstacles, regions)
            if config_cost:
                nonnavigable.append(config_cost[0])
                unsafe.append(config_cost[1])
                i += 1
        
        unsafe = sum(unsafe)
        efficiency = len(regions)
        nonnavigable_mu = mean(nonnavigable)
        nonnavigable_var = variance(nonnavigable)
        
        criterion = np.array([unsafe, efficiency, nonnavigable_mu, nonnavigable_var])
        weights = np.array((12, 2, 25, 25))
        problem_cost = np.sum(criterion * weights)
        return problem_cost

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