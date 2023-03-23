import math
import time
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from math import pi, inf
from scipy import optimize
from itertools import product
from statistics import mean, variance
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

class Language:
    def __init__(self, args):
        corners = list(product((1, -1), repeat=2))
        self.configs_to_consider = 100
        self.num_obstacles = args.num_obs
        self.obs_constr = args.obs_constr
        self.goal_constr = args.goal_constr
        self.start_constr = args.start_constr
        self.obs_area = (pi*args.obs_size) ** 2
        self.square = Polygon([corners[0], corners[2], corners[3], corners[1]])
        self.boundaries = [LineString([corners[0], corners[2]]),
                           LineString([corners[2], corners[3]]),
                           LineString([corners[3], corners[1]]),
                           LineString([corners[1], corners[0]])]
    
    # Both endpoints must be on an environment boundary to be considered valid
    def _get_valid_lines(self, lines):
        valid_lines = [*self.boundaries]
        # Get valid lines s.t. both endpoints are on an environment boundary
        for line in lines:
            intersection = self.square.intersection(line)
            if not intersection.is_empty and np.any(np.abs([*intersection.coords]) == 1, axis=1).all():
                valid_lines.append(intersection)
        
        return valid_lines        
            
    # Create polygonal regions from lines
    def _create_regions(self, lines):
        valid_lines = self._get_valid_lines(lines)
        lines = MultiLineString(valid_lines)
        lines = lines.buffer(distance=1e-12)
        boundary = lines.convex_hull
        polygons = boundary.difference(lines)
        regions = [polygons] if polygons.geom_type == 'Polygon' else \
            [polygons.geoms[i] for i in range(len(polygons.geoms))]
        return regions
    
    # Get info on neighboring regions
    def _find_neighbors(self, cur_idx, goal_idx, obstacles, regions):
        neighbors = {}
        cur_region = regions[cur_idx]
        goal_region = regions[goal_idx]
        for neighbor in regions:
            if not cur_region.equals_exact(neighbor, 0) and cur_region.dwithin(neighbor, 2e-12):
                idx = regions.index(neighbor)
                g = cur_region.centroid.distance(neighbor.centroid)
                h = neighbor.centroid.distance(goal_region.centroid)
                f = g + h
                is_goal = neighbor.equals(goal_region)
                is_safe = not any(neighbor.contains(obstacles))
                neighbors[idx] = (is_goal, is_safe, f)

        return neighbors
    
    # Get next region to move to
    def _get_next_region(self, prev_idx, neighbors):
        min_f = inf
        next_idx = None
        if prev_idx in neighbors: del neighbors[prev_idx]

        for idx, neighbor in neighbors.items():
            if neighbor[0]:
                next_idx = idx
                break
            elif neighbor[1] and neighbor[2] < min_f:
                min_f = neighbor[2]
                next_idx = idx

        return next_idx
    
    # Check if there exists a safe path from start to goal
    def _get_safe_path(self, start_idx, goal_idx, obstacles, regions):
        safe_path = True
        
        prev_idx = None
        cur_idx = start_idx
        while cur_idx != goal_idx:
            neighbors = self._find_neighbors(start_idx, goal_idx, obstacles, regions)
            next_idx = self._get_next_region(prev_idx, neighbors)
            if not next_idx:
                safe_path = False
                break
            prev_idx = cur_idx
            cur_idx = next_idx
            
        return safe_path
    
    """
    Calculate cost of a configuration (i.e. start, goal, and obstacles)
    with respect to the regions and positions under some positional constraints: 
        1. Unsafe area caused by obstacles
        2. Unsafe plan caused by non-existent path from start to goal while avoiding unsafe area
    """
    def _config_cost(self, start_idx, goal_idx, obstacles, regions):
        obs_idx = list(set([idx for idx, region in enumerate(regions) 
                            for obs in obstacles if region.contains(obs)]))
        nonnavigable = sum([regions[idx].area for idx in obs_idx])
        
        if start_idx in obs_idx or goal_idx in obs_idx:
            unsafe = True
        elif start_idx == goal_idx:
            unsafe = False
        else:
            safe = self._get_safe_path(start_idx, goal_idx, obstacles, regions)
            unsafe = not safe
            
        return nonnavigable, unsafe
        
    """ 
    Calculate cost of a given problem (i.e. all configurations) 
    with respect to the regions and the given positional constraints: 
        1. Unsafe plans
        2. Language efficiency
        3. Mean of nonnavigable area
        4. Variance of nonnavigable area
    """
    def _optimizer(self, lines):
        lines = [LineString([tuple(lines[i:i+2]), tuple(lines[i+2:i+4])]) 
                 for i in range(0, len(lines), 4)]
        regions = self._create_regions(lines)
        if len(regions) == 0: return math.inf
        
        nonnavigable, unsafe = [], []
        for _ in range(self.configs_to_consider):
            start = Point(random.choice(self.start_constr)*np.random.uniform(0, 1, size=(2,)))
            goal = Point(random.choice(self.goal_constr)*np.random.uniform(0, 1, size=(2,)))
            
            start_idx = list(map(lambda region: region.contains(start), regions)).index(True)
            goal_idx = list(map(lambda region: region.contains(goal), regions)).index(True)
            
            obs_pos = np.random.uniform(0, 1, size=(self.num_obstacles, 2))
            obstacles = [Point(random.choice(self.obs_constr)*obs_pos[i]) for i in range(self.num_obstacles)]
            
            config_cost = self._config_cost(start_idx, goal_idx, obstacles, regions)
            nonnavigable += [config_cost[0]]
            unsafe += [config_cost[1]]
        
        unsafe = sum(unsafe)
        efficiency = len(regions)
        nonnavigable_mu = mean(nonnavigable)
        nonnavigable_var = variance(nonnavigable)
        
        criterion = np.array([unsafe, efficiency, nonnavigable_mu, nonnavigable_var])
        weights = np.array((1, 3, 15, 25))
        problem_cost = np.sum(criterion * weights)
        return problem_cost

    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_lines(self):
        lb, ub = -1.25, 1.25
        optim_val, optim_lines = math.inf, None
        start = time.time()
        for num in range(2, 10):
            print(f'Generating langauge with {num} lines...')
            bounds = [(lb, ub) for _ in range(num*4)]
            res = optimize.differential_evolution(self._optimizer, bounds,
                                                  init='sobol')
            if optim_val > res.fun:
                optim_val = res.fun
                optim_lines = res.x
        
        end = time.time()
        print(f'Optimization time: {end-start:.2f} seconds')
        optim_lines = np.reshape(optim_lines, (-1, 4))
        return optim_lines
    
    # Visualize regions that define the language
    def _visualize(self, regions):
        for idx, region in enumerate(regions):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        plt.savefig('regions.png')
    
    # Returns regions that define the language
    def create(self):
        lines = self._generate_optimal_lines()
        lines = [LineString([line[0:2], line[2:4]]) for line in lines]
        regions = self._create_regions(lines)
        self._visualize(regions)
        return regions