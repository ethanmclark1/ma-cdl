# This code makes use of the DEAP (Distributed Evolutionary Algorithms in Python) library:
# Félix-Antoine Fortin et al. (2021). DEAP: Evolutionary Algorithms Made Easy. GitHub.
# Available at: https://github.com/DEAP/deap

import time
import wandb
import numpy as np

from languages.utils.cdl import CDL
from deap import base, creator, tools

"""Using Genetic Algorithm with Elitism"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self._init_hyperparams()
        self._init_deap()

    def _init_hyperparams(self):
        self.mut_prob = 0.30
        self.cross_prob = 0.70
        self.population_size = 100
        self.num_generations = 150
        self.num_elites = round(self.population_size * 0.04)
        self.tournament_size = round(self.population_size * 0.05)

    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'EA/{problem_instance.capitalize()}')
        config = wandb.config
        config.weights = self.weights
        config.configs_to_consider = self.configs_to_consider
        
        config.mut_prob = self.mut_prob
        config.cross_prob = self.cross_prob
        config.num_elites = self.num_elites
        config.population_size = self.population_size
        config.tournament_size = self.tournament_size
        config.num_generations = self.num_generations
        
    # Upload regions to Weights and Biases
    def _log_regions(self, problem_instance, num_lines, coeffs, cost):
        coeffs = np.reshape(coeffs, (-1, self.action_dim))
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
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", self.cxTwoPoint)
        self.toolbox.register("mutate", self.mutRandFloat)
    
    # Set problem instance for optimizer
    def _set_instance(self, problem_instance):
        if hasattr(self.toolbox, 'evaluate'):
            self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", self.optimizer, problem_instance=problem_instance)
    
    def _attr_float(self, num_lines):
        coeffs = self.rng.choice(self.possible_coeffs, size=num_lines)
        return coeffs.flatten()
    
    # Set number of lines for population
    def _set_num_lines(self, num_lines):
        if hasattr(self.toolbox, 'individual'):
            self.toolbox.unregister('attr_float')
            self.toolbox.unregister("individual")
            self.toolbox.unregister("population")
        self.toolbox.register("attr_float", self._attr_float, num_lines)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.population_size)

    def optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super().optimizer(regions, problem_instance)
        return (scenario_cost,)
    
    # Perform Two-Point Crossover on the offspring
    def cxTwoPoint(self, ind1, ind2):
        size = len(ind1)
        cxpoint1 = self.rng.integers(0, size)
        cxpoint2 = self.rng.integers(0, size)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
            
        alleles1, alleles2 = ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2]
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = alleles2, alleles1

        ind1_mapped = CDL.get_mapped_coeffs(ind1)
        ind2_mapped = CDL.get_mapped_coeffs(ind2)
        flattened_ind1 = np.array(ind1_mapped).flatten()
        flattened_ind2 = np.array(ind2_mapped).flatten()
        ind1 = creator.Individual(flattened_ind1)
        ind2 = creator.Individual(flattened_ind2)
        
        return ind1, ind2
    
    # Generate random floating point number
    def mutRandFloat(self, individual):
        size = len(individual)
        mutated_allele = self.rng.integers(0, size)
        mutation = self.rng.uniform(-0.1, 0.1)
        individual[mutated_allele] = mutation

        ind_mapped = CDL.get_mapped_coeffs(individual)
        flattened_ind = np.array(ind_mapped).flatten()
        individual = creator.Individual(flattened_ind)
        return individual
    
    # Perform mutation and crossover on the offspring
    def evolve(self, population, toolbox):
        offspring = [toolbox.clone(ind) for ind in population]

        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if self.rng.random() < self.cross_prob:
                offspring[i-1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
                
        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if self.rng.random() < self.mut_prob:
                offspring[i] = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring
    
    # Runs the genetic algorithm with elitism
    def eaSimpleWithElitism(self, population, stats, halloffame, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for gen in range(1, self.num_generations + 1):
            offspring = self.toolbox.select(population, len(population) - self.num_elites)

            offspring = self.evolve(offspring, self.toolbox)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
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
        self._init_wandb(problem_instance)
        self._set_instance(problem_instance)
        
        best_val = float('-inf')
        best_coeffs = None
        best_num_lines = None
        
        start_time = time.time()
        for num_lines in range(self.min_lines, self.max_lines + 1):
            self._set_num_lines(num_lines)
            population = self.toolbox.population()
            hof = tools.HallOfFame(1, similar=np.array_equal)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)

            self.eaSimpleWithElitism(population, stats, hof)
            
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
        
        best_coeffs = np.array(best_coeffs).reshape(-1, self.action_dim)
        return best_coeffs