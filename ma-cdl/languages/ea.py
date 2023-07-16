import time
import wandb
import random
import numpy as np

from languages.utils.cdl import CDL
from deap import base, creator, tools, algorithms

"""Using Genetic Algorithm"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        random.seed(42)
        self._init_hyperparams()
        self._init_deap()

    def _init_hyperparams(self):
        self.mutation_prob = 0.2
        self.crossover_prob = 0.6
        self.tournament_size = 3
        self.population_size = 100
        self.num_generations = 150

    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='EA')
        config = wandb.config
        config.weights = self.weights
        config.resolution = self.resolution
        config.configs_to_consider = self.configs_to_consider
        
        config.mutation_prob = self.mutation_prob
        config.crossover_prob = self.crossover_prob
        config.population_size = self.population_size
        config.tournament_size = self.tournament_size
        config.num_generations = self.num_generations
        
    def attr_float(self):
        value = self.rng.uniform(-1, 1)
        discretized_value = round(value / self.resolution) * self.resolution
        return discretized_value
        
    def _init_deap(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.attr_float)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=0.05)
        
    def _set_instance(self, problem_instance):
        if hasattr(self.toolbox, 'evaluate'):
            self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", self.optimizer, problem_instance=problem_instance)
        
    def _set_num_lines(self, num_lines):
        if hasattr(self.toolbox, 'individual'):
            self.toolbox.unregister("individual")
            self.toolbox.unregister("population")
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=3*num_lines)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, num_lines, coeffs, cost):
        coeffs = np.reshape(coeffs, (-1, 3))
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        pil_image = super()._get_image(problem_instance, 'Lines', num_lines, regions, -cost)
        wandb.log({"image": wandb.Image(pil_image)})

    def optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super().optimizer(regions, problem_instance)
        return (scenario_cost,)
    
    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()
        self._set_instance(problem_instance)
        
        best_val = float('inf')
        best_coeffs = None
        best_num_lines = None
        
        start_time = time.time()
        for num_lines in range(self.min_lines, self.max_lines + 1):
            self._set_num_lines(num_lines)
            pop = self.toolbox.population(n=self.population_size)
            hof = tools.HallOfFame(1, similar=np.array_equal)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)

            pop, logbook = algorithms.eaSimple(pop, self.toolbox, self.crossover_prob, 
                                               self.mutation_prob, self.num_generations, 
                                               stats, hof)
            
            optim_coeffs = hof[0]
            optim_val = hof[0].fitness.values[0]
            if optim_val < best_val:
                best_val = optim_val
                best_coeffs = optim_coeffs
                best_num_lines = num_lines
            
            wandb.log({"Optimal Coeffs": optim_coeffs})
            self._log_regions(problem_instance, num_lines, optim_coeffs, -optim_val)
            
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
            
        wandb.log({"Final Reward": -best_val})
        wandb.log({"Best Coeffs": best_coeffs})
        wandb.log({"Best Num Lines": best_num_lines})
        wandb.log({"Elapsed Time": elapsed_time})
        wandb.finish()
        
        best_coeffs = np.reshape(best_coeffs, (-1, 3))
        return best_coeffs