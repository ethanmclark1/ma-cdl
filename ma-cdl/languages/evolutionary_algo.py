import os
import time
import pickle
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
        
    def _save(self, class_name, scenario):
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}-{scenario}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(self.language, file)
    
    def _load(self, class_name, scenario):
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}-{scenario}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        self.language = language
    
    def _optimizer(self, coeffs, scenario):
        problem_cost, _ = super()._optimizer(coeffs, scenario)
        return problem_cost
        
    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, scenario):
        lb, ub = -1, 1
        optim_val, optim_coeffs = inf, None
        start = time.time()
        for num in range(1, 7):
            bounds = [(lb, ub) for _ in range(3*num)]
            res = optimize.differential_evolution(self._optimizer, bounds, args=(scenario,),
                                                  maxiter=100*num, init='sobol')
            print(f'Cost: {res.fun}')
            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x
        
        end = time.time()
        print(f'Elapsed time: {end - start} seconds')
        optim_coeffs = np.reshape(optim_coeffs, (-1, 3))
        return optim_coeffs

    # Returns regions that defines the language
    def get_language(self, scenario):
        class_name = self.__class__.__name__
        try:
            self._load(class_name, scenario)
        except:
            print(f'No stored {class_name} language for {scenario} problem.')
            print('Generating new language...')
            coeffs = self._generate_optimal_coeffs(scenario)
            lines = self._get_lines_from_coeffs(coeffs)
            self.language = self._create_regions(lines)
            self._save(class_name, scenario)
        
        self._visualize(class_name, scenario)