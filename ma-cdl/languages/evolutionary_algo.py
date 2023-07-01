import wandb
import numpy as np

from math import inf
from scipy import optimize
from languages.utils.cdl import CDL

"""Evolutionary Algorithm"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self._init_hyperparams()

    def _init_hyperparams(self):
        self.tol = 0.04
        self.max_iter = 175

    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='EA')
        config = wandb.config
        config.tol = self.tol
        config.weights = self.weights
        config.max_iter = self.max_iter

    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, num_lines, coeffs, cost):
        coeffs = np.reshape(coeffs, (-1, 3))
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        pil_image = super()._get_image(problem_instance, 'num_lines', num_lines, regions, -cost)
        wandb.log({"image": wandb.Image(pil_image)})

    def optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super().optimizer(regions, problem_instance)
        return scenario_cost

    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()

        lb, ub = -1, 1
        optim_val, optim_coeffs = inf, None
        for num in range(self.min_lines, self.max_lines + 1):
            bounds = [(lb, ub) for _ in range(3*num)]
            res = optimize.differential_evolution(self.optimizer, bounds, args=(problem_instance,),
                                                  tol=self.tol, maxiter=self.max_iter*num, init='sobol')
            self._log_regions(problem_instance, num, res.x, res.fun)

            if optim_val > res.fun:
                optim_val = res.fun
                optim_coeffs = res.x

        wandb.finish()
        optim_coeffs = np.reshape(optim_coeffs, (-1, 3))
        return optim_coeffs
