# This code makes use of the DEAP (Distributed Evolutionary Algorithms in Python) library:
# Félix-Antoine Fortin et al. (2021). DEAP: Evolutionary Algorithms Made Easy. GitHub.
# Available at: https://github.com/DEAP/deap

import time
import wandb
import random
import numpy as np

from languages.utils.cdl import CDL
from deap import base, creator, tools, algorithms

"""Using Genetic Algorithm with Elitism"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        random.seed(42)
        self._init_hyperparams()
        self._init_deap()

    def _init_hyperparams(self):
        self.mutation_prob = 0.3
        self.crossover_prob = 0.70
        self.population_size = 150
        self.num_generations = 150
        self.num_elites = round(self.population_size * 0.03)
        self.tournament_size = round(self.population_size * 0.05)

    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='EA')
        config = wandb.config
        config.weights = self.weights
        config.configs_to_consider = self.configs_to_consider
        
        config.mutation_prob = self.mutation_prob
        config.crossover_prob = self.crossover_prob
        config.population_size = self.population_size
        config.tournament_size = self.tournament_size
        config.num_generations = self.num_generations
        
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, num_lines, coeffs, cost):
        coeffs = np.reshape(coeffs, (-1, 3))
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        pil_image = super()._get_image(problem_instance, 'Lines', num_lines, regions, cost)
        wandb.log({"image": wandb.Image(pil_image)})

    # Initialize requirements for GA
    def _init_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.rng.uniform, -1, 1)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=0.05)
    
    # Set problem instance for optimizer
    def _set_instance(self, problem_instance):
        if hasattr(self.toolbox, 'evaluate'):
            self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", self.optimizer, problem_instance=problem_instance)
    
    # Set number of lines for population
    def _set_num_lines(self, num_lines):
        if hasattr(self.toolbox, 'individual'):
            self.toolbox.unregister("individual")
            self.toolbox.unregister("population")
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=3*num_lines)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super().optimizer(regions, problem_instance)
        return (scenario_cost,)
    
    # Runs the genetic algorithm with elitism
    def eaSimpleWithElitism(self, population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for gen in range(1, ngen + 1):
            offspring = toolbox.select(population, len(population) - self.num_elites)

            # Perform mutation and crossover on the offspring
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            elites = tools.selBest(population, self.num_elites)

            population[:] = offspring + elites

            halloffame.update(population)

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_coeffs(self, problem_instance):
        self._init_wandb()
        self._set_instance(problem_instance)
        
        best_val = float('-inf')
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

            self.eaSimpleWithElitism(pop, self.toolbox, self.crossover_prob, 
                                     self.mutation_prob, self.num_generations, 
                                     stats, hof, verbose=False)
            
            optim_coeffs = hof[0]
            optim_val = hof[0].fitness.values[0]
            if optim_val > best_val:
                best_val = optim_val
                best_coeffs = optim_coeffs
                best_num_lines = num_lines
            
            wandb.log({"Optimal Coeffs": optim_coeffs})
            self._log_regions(problem_instance, num_lines, optim_coeffs, optim_val)
            
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 3600, 3)
            
        wandb.log({"Final Reward": best_val})
        wandb.log({"Best Coeffs": best_coeffs})
        wandb.log({"Best Num Lines": best_num_lines})
        wandb.log({"Elapsed Time": elapsed_time})
        wandb.finish()
        
        best_coeffs = np.array(best_coeffs).reshape(-1, 3)
        return best_coeffs