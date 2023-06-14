import time
import wandb
import numpy as np

from math import inf
from scipy import optimize
from languages.utils.cdl import CDL

"""Evolutionary Algorithm"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='autoencoder')
        config = wandb.config
        config.learning_rate = self.learning_rate
        config.num_train_epochs = self.num_train_epochs
                
    def _optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super()._optimizer(regions, problem_instance)
        return scenario_cost
        
    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, problem_instance):
        lb, ub = -1, 1
        optim_val, optim_coeffs = inf, None
        for num in range(self.min_lines, self.max_lines):
            bounds = [(lb, ub) for _ in range(3*num)]
            res = optimize.differential_evolution(self._optimizer, bounds, args=(problem_instance,),
                                                  maxiter=100*num, init='sobol')
            wandb.log({'cost': res.fun})
            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x
        
        optim_coeffs = np.reshape(optim_coeffs, (-1, 3))
        return optim_coeffs