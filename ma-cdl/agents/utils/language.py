import math
import time
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
        self.configs_to_consider = 30
        self.num_obstacles = args.num_obstacles
        self.num_languages = args.num_languages
        self.obs_area = (math.pi*args.obstacle_size) ** 2
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
    Cost function to minimize:
        1. Mean and variance of collision probability
        2. Mean and variance of non-navigable area
        3. Variance of region area 
        4. Language efficiency
    """
    def _optimizer(self, lines):
        lines = [LineString([tuple(lines[i:i+2]), tuple(lines[i+2:i+4])]) 
                 for i in range(0, len(lines), 4)]
        regions = self._create_regions(lines)
        if len(regions) == 0: return math.inf
        
        collision, unsafe = [], []
        # Obstacle(s) are constrained to be in the top right quadrant
        for _ in range(self.configs_to_consider):
            p_obs, p_region, p_obs_and_region = 0, 0, 0
            conditional_prob, nonnavigable = 0, 0
            obs_pos = np.random.rand(self.num_obstacles, 2)
            obs_list = [Point(obs_pos[i]) for i in range(self.num_obstacles)]
            
            for obs, region in product(obs_list, regions):
                if region.contains(obs):
                    p_obs += (self.obs_area / self.square.area)
                    p_region += (region.area / self.square.area)
                    p_obs_and_region += (p_obs * p_region)
                    conditional_prob += (p_obs_and_region / p_region)
                    nonnavigable += region.area
            collision.append(conditional_prob)
            unsafe.append(nonnavigable)
        
        collision_mu = mean(collision)
        collision_var = variance(collision)
        unsafe_mu = mean(unsafe)
        unsafe_var = variance(unsafe)
        region_var = 0 if len(regions) == 1 else variance([region.area for region in regions])
        efficiency = len(regions)
        
        criterion = np.array([collision_mu, collision_var, unsafe_mu, unsafe_var, region_var, efficiency])
        weights = np.array((15, 17, 9, 18, 10, 2))
        cost = np.sum(criterion * weights)
        return cost

    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_lines(self):
        lb, ub = -1.25, 1.25
        optim_val, optim_lines = math.inf, None
        start = time.time()
        for num in range(2, self.num_languages+2):
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