import math
import time
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from itertools import product
from statistics import mean, variance
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

class Language:
    def __init__(self, args):
        corners = list(product((1, -1), repeat=2))
        self.configs_to_consider = 100
        self.start_constr = args.start_constr
        self.goal_constr = args.goal_constr
        self.obs_constr = args.obs_constr
        self.num_obstacles = args.num_obs
        self.obs_area = (math.pi*args.obs_size) ** 2
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
    
    """
    Cost function to minimize for current configuration:
        1. Unsafe area
        2. Unsafe plans    
    """
    def _config_cost(self, start, goal, obstacles, regions):
        unsafe_area = sum(map(lambda obstacle: regions[obstacle].area, obstacles))
        
        if start is obstacles or goal in obstacles:
            unsafe_plan = True
        else:
            a=3        
        
        return unsafe_area, unsafe_plan
        
        
    """ 
    Cost function to minimize for entire language (i.e. all configurations):
        1. Mean and variance of unsafe area
        2. Variance of region area 
        3. Unsafe plans
        4. Language efficiency
    """
    def _optimizer(self, lines):
        lines = [LineString([tuple(lines[i:i+2]), tuple(lines[i+2:i+4])]) 
                 for i in range(0, len(lines), 4)]
        regions = self._create_regions(lines)
        if len(regions) == 0: return math.inf
        
        unsafe_area, unsafe_plan = [], []
        # Obstacle(s) are constrained to be in the top right quadrant
        for _ in range(self.configs_to_consider):
            nonnavigable = 0
            start = Point(random.choice(self.start_constr)*np.random.uniform(0, 1, size=(2,)))
            goal = Point(random.choice(self.goal_constr)*np.random.uniform(0, 1, size=(2,)))
            
            start_region = list(map(lambda region: region.contains(start), regions)).index(True)
            goal_region = list(map(lambda region: region.contains(start), regions)).index(True)
            
            obs_pos = np.random.uniform(0, 1, size=(self.num_obstacles, 2))
            obs_list = [Point(random.choice(self.obs_constr)*obs_pos[i]) for i in range(self.num_obstacles)]
            obs_region = list(set([idx for idx, region in enumerate(regions) 
                                   for obs in obs_list if region.contains(obs)]))
            
            config_safety = self._config_cost(start_region, goal_region, obs_region, regions)
            unsafe_area += config_safety[0]
            unsafe_plan += config_safety[1]
        
        unsafe_mu = mean(unsafe_area)
        unsafe_var = variance(unsafe_plan)
        region_var = 0 if len(regions) == 1 else variance([region.area for region in regions])
        unsafe_plan = sum(unsafe_plan)
        efficiency = len(regions)
        
        criterion = np.array([unsafe_mu, unsafe_var, region_var, unsafe_plan, efficiency])
        weights = np.array((9, 18, 10, 2))
        cost = np.sum(criterion * weights)
        return cost

    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_lines(self):
        lb, ub = -1.25, 1.25
        optim_val, optim_lines = math.inf, None
        start = time.time()
        for num in range(2, 3):
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