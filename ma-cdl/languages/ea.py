# This code makes use of the DEAP (Distributed Evolutionary Algorithms in Python) library:
# Félix-Antoine Fortin et al. (2021). DEAP: Evolutionary Algorithms Made Easy. GitHub.
# Available at: https://github.com/DEAP/deap

import time
import math
import wandb
import numpy as np
import matplotlib.pyplot as plt

from languages.utils.cdl import CDL
from deap import base, creator, tools

"""Using Genetic Algorithm with Elitism"""
class EA(CDL):
    def __init__(self, scenario, world):
        super().__init__(scenario, world)
        self._init_hyperparams()
        self._init_deap()
        
        self.rng = np.random.default_rng()

    def _init_hyperparams(self):
        self.alpha = 1.5
        self.sigma = 0.65
        self.num_elites = 2
        self.mut_prob = 0.30
        self.cross_prob = 0.70
        self.tournament_size = 3
        self.population_size = 5
        self.num_generations = 100
        
    def _init_wandb(self, problem_instance):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name=f'EA/{problem_instance.capitalize()}')
        config = wandb.config
        config.configs_to_consider = self.configs_to_consider
        
        config.alpha = self.alpha
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
        self.toolbox.register("select", tools.selTournament)
        self.toolbox.register("mate", self.cxTwoPoint)
        self.toolbox.register("mutate", self.mutRandFloat)
        self.toolbox.register('distance', self.calc_distance)
    
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
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    
    # Calculate similarity between two individuals (coefficients) using euclidean distance on slope and intercept of lines
    def calc_distance(self, ind1, ind2):
        mapped_ind1 = np.array(CDL.get_mapped_coeffs(ind1))
        mapped_ind2 = np.array(CDL.get_mapped_coeffs(ind2))
        total_dist = 0
        
        for m_ind1 in mapped_ind1:
            for m_ind2 in mapped_ind2:
                if m_ind1[0] != 0 and m_ind1[1] != 0 and m_ind2[0] != 0 and m_ind2[1] != 0:
                    # Both lines are diagonal
                    m1 = -m_ind1[1] / m_ind1[0]
                    c1 = -m_ind1[2] / m_ind1[0]
                    m2 = -m_ind2[1] / m_ind2[0]
                    c2 = -m_ind2[2] / m_ind2[0]
                    dist = abs(m1 - m2) + abs(c1 - c2)
                elif m_ind1[0] == 0 and m_ind1[1] != 0 and m_ind2[0] == 0 and m_ind2[1] != 0:
                    # Both lines are horizontal
                    c1 = -m_ind1[2] / m_ind1[1]
                    c2 = -m_ind2[2] / m_ind2[1]
                    dist = abs(c1 - c2)
                elif m_ind1[1] == 0 and m_ind1[0] != 0 and m_ind2[1] == 0 and m_ind2[0] != 0:
                    # Both lines are vertical
                    c1 = -m_ind1[2] / m_ind1[0]
                    c2 = -m_ind2[2] / m_ind2[0]
                    dist = abs(c1 - c2)
                else:
                    # Lines are not comparable (i.e., one is vertical and the other is horizontal)
                    dist = 6
                total_dist += dist
                
        num_comparisons = (len(ind1) // 3) ** 2
        avg_dist = total_dist / num_comparisons
        
        return avg_dist

    def optimizer(self, coeffs, problem_instance):
        lines = CDL.get_lines_from_coeffs(coeffs)
        valid_lines = CDL.get_valid_lines(lines)
        regions = CDL.create_regions(valid_lines)
        scenario_cost = super().optimizer(regions, problem_instance)
        return (scenario_cost + 100,) # Add 100 to avoid negative fitness values
    
    # Sigma represents the minimum distance between individuals and alpha represents the rate of decay
    def fitness_sharing(self, ind, ind_fit, population):
        def sharing_function(individual, other):
            distance = self.toolbox.distance(individual, other)            
            return 1 - (distance / self.sigma) ** self.alpha if distance < modified_sigma else 0
        
        modified_sigma = self.sigma * ((len(ind) // 3))
        pop = [p for p in population if not np.array_equal(p, ind)]
        sum_sharing = sum([sharing_function(ind, other) for other in pop])
        adjusted_fitness = ind_fit / sum_sharing if sum_sharing != 0 else ind_fit
        return (adjusted_fitness,)

    # Penalizes individuals that are similar to increase diversity
    def calculate_shared_fitness(self, population):
        raw_fitnesses = [ind.fitness.values[0] for ind in population]
        
        for i, ind in enumerate(population):
            ind.fitness.values = self.fitness_sharing(ind, raw_fitnesses[i], population)
    
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
        
        if mutated_allele % 3 == 2:
            possible_vals = np.arange(-18, 18 + self.granularity, self.granularity)
            mutation = self.rng.choice(possible_vals) / 100
        else:
            possible_vals = np.arange(-10, 10 + 10, 10)
            mutation = self.rng.choice(possible_vals) / 100
        individual[mutated_allele] = mutation

        ind_mapped = CDL.get_mapped_coeffs(individual)
        flattened_ind = np.array(ind_mapped).flatten()
        individual = creator.Individual(flattened_ind)
        return individual
    
    # Perform mutation and crossover on the offspring
    def evolve(self, population):
        offspring = [self.toolbox.clone(ind) for ind in population]

        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if self.rng.random() < self.cross_prob:
                offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
                
        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if self.rng.random() < self.mut_prob:
                offspring[i] = self.toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring

    # Runs a genetic algorithm with elitism and fitness sharing
    def genetic_algorithm(self, population, stats, halloffame, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        self.calculate_shared_fitness(population)
        
        halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for gen in range(1, self.num_generations + 1):
            offspring = self.toolbox.select(population, len(population) - self.num_elites, self.tournament_size)

            offspring = self.evolve(offspring)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.calculate_shared_fitness(offspring)

            elites = tools.selBest(population, self.num_elites)

            population[:] = offspring + elites

            halloffame.update(population)

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
                
        return population, logbook, halloffame

    # # Minimizes cost function to generate the optimal lines
    # def _generate_optimal_coeffs(self, problem_instance):
    #     self._init_wandb(problem_instance)
    #     self._set_instance(problem_instance)
        
    #     best_val = float('-inf')
    #     best_coeffs = None
    #     best_num_lines = None
        
    #     start_time = time.time()
    #     for num_lines in range(self.min_lines, self.max_lines + 1):
    #         self._set_num_lines(num_lines)
    #         population = self.toolbox.population()
    #         hof = tools.HallOfFame(1, similar=np.array_equal)
            
    #         stats = tools.Statistics(lambda ind: ind.fitness.values)
    #         stats.register("avg", np.mean)
    #         stats.register("min", np.min)
    #         stats.register("max", np.max)
    #         stats.register("diversity", np.std)
            
    #         population, _, hof = self.genetic_algorithm(population, stats, hof)
            
    #         optim_coeffs = hof[0]
    #         optim_val = hof[0].fitness.values[0]
    #         if optim_val > best_val:
    #             best_val = optim_val
    #             best_coeffs = optim_coeffs
    #             best_num_lines = num_lines
            
    #         wandb.log({"Optimal Coeffs": optim_coeffs})
    #         wandb.log({"Diversity": stats.compile(population)["diversity"]})
    #         self._log_regions(problem_instance, num_lines, optim_coeffs, optim_val)
            
    #     end_time = time.time()
    #     elapsed_time = round((end_time - start_time) / 3600, 3)
            
    #     wandb.log({"Final Reward": best_val})
    #     wandb.log({"Best Coeffs": best_coeffs})
    #     wandb.log({"Best Num Lines": best_num_lines})
    #     wandb.log({"Elapsed Time": elapsed_time})
    #     wandb.finish()
        
    #     best_coeffs = np.array(best_coeffs).reshape(-1, self.action_dim)
    #     return best_coeffs
    def _generate_optimal_coeffs(self, problem_instance):
        problem_instance = 'corners'
        self._set_instance(problem_instance)
        self._set_num_lines(4)
             
        num_searches = 50
        best_val = float('-inf')
        
        hyperparameters = {
            'alpha': np.arange(0.5, 2.0, 0.1),
            'mutation_prob': np.arange(0.1, 0.6, 0.05),
            'cross_prob': np.arange(0.1, 0.6, 0.05),
            'num_elites': list(range(1, 8)),
            'tournament_size': list(range(1, 8)),
            'population_size': list(range(100, 150 + 5, 5)),
            'num_generations': list(range(100, 200 + 5, 5)),
        }
        
        for _ in range(num_searches):
            parameters = {key: self.rng.choice(value) for key, value in hyperparameters.items()}
            self.alpha = parameters['alpha']
            self.mut_prob = parameters['mutation_prob']
            self.cross_prob = parameters['cross_prob']
            self.num_elites = parameters['num_elites']
            self.tournament_size = parameters['tournament_size']
            self.population_size = parameters['population_size']
            self.num_generations = parameters['num_generations']
            self._init_wandb(problem_instance)
            
            population = self.toolbox.population(n=self.population_size)
            hof = tools.HallOfFame(1, similar=np.array_equal)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            stats.register("diversity", np.std)
            
            population, _, hof = self.genetic_algorithm(population, stats, hof, verbose=False)
            
            optim_coeffs = hof[0]
            optim_val = hof[0].fitness.values[0]
            self._log_regions('corners', 4, optim_coeffs, optim_val)
            wandb.log({"Diversity": stats.compile(population)["diversity"]})
            wandb.finish()
            
            if optim_val > best_val:
                best_val = optim_val
                best_hyperparameters = parameters
                
        print('Best Hyperparameters:', best_hyperparameters)
        print('Best Value:', best_val)