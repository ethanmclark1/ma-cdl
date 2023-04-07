import time
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from math import pi, inf
from scipy import optimize
from shapely import points
from itertools import product
from statistics import mean, variance
from agents.utils.path_finder.rrt_star import RRTStar
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')
weights = np.array((15, 2, 35, 35))

class Language:
    def __init__(self, env):
        self.env = env
        self.configs_to_consider = 2
        self.num_obstacles = self.env.metadata['num_obstacles']
        self.problem_type = env.unwrapped.world.problem_type
        self.start_constr = env.unwrapped.world.start_constr
        self.goal_constr = env.unwrapped.world.goal_constr
        self.obs_constr = env.unwrapped.world.obs_constr
        self.rrt_star = RRTStar(env.metadata["agent_radius"], env.metadata["obstacle_radius"])
        
        corners = list(product((1, -1), repeat=2))
        self.square = Polygon([corners[0], corners[2], corners[3], corners[1]])
        self.boundaries = [LineString([corners[0], corners[2]]),
                           LineString([corners[2], corners[3]]),
                           LineString([corners[3], corners[1]]),
                           LineString([corners[1], corners[0]])]
        
    def _save(self, regions):
        with open(f'ma-cdl/agents/utils/stored_langs/{self.problem_type}+{weights}.pkl', 'wb') as f:
            pickle.dump(regions, f)
    
    def load(self):
        with open(f'ma-cdl/agents/utils/stored_langs/{self.problem_type}+{weights}.pkl', 'rb') as f:
            regions = pickle.load(f)
        return regions
    
    def _get_valid_lines(self, lines):
        valid_lines = list(self.boundaries)

        for line in lines:
            intersection = self.square.intersection(line)
            if not intersection.is_empty:
                coords = np.array(intersection.coords)
                if np.any(np.abs(coords) == 1, axis=1).all():
                    valid_lines.append(intersection)

        return valid_lines    
            
    # Create polygonal regions from lines
    def _create_regions(self, lines):
        valid_lines = self._get_valid_lines(lines)
        lines = MultiLineString(valid_lines).buffer(distance=1e-12)
        boundary = lines.convex_hull
        polygons = boundary.difference(lines)
        regions = [polygons] if polygons.geom_type == 'Polygon' else [polygons.geoms[i] for i in range(len(polygons.geoms))]
        return regions
    
    def _generate_points(self):
        obstacles = []    
        start = np.random.uniform(*zip(*self.start_constr))
        goal = np.random.uniform(*zip(*self.goal_constr))
        
        # set state of obstacles
        if isinstance(self.obs_constr, tuple):
            obstacles = [np.random.uniform(*zip(*self.obs_constr)) for _ in range(self.num_obstacles)]
        else:
            obstacles = [np.random.uniform(*zip(*self.obs_constr[0])) for _ in range(self.num_obstacles // 2)]
            obstacles += [np.random.uniform(*zip(*self.obs_constr[1])) for _ in range(self.num_obstacles // 2)]
        
        return start, goal, obstacles
    
    """
    Calculate cost of a configuration (i.e. start, goal, and obstacles)
    with respect to the regions and positions under some positional constraints: 
        1. Unsafe area caused by obstacles
        2. Unsafe plan caused by non-existent path from start to goal while avoiding unsafe area
    """
    def _config_cost(self, start, goal, obstacles, regions):
        obstacles_idx = set(idx for idx, region in enumerate(regions)
                            for obs in obstacles if region.contains(Point(obs)))
        nonnavigable = sum(regions[idx].area for idx in obstacles_idx)
        
        try:
            path = self.rrt_star.plan(start, goal, obstacles)
            path = points(path)
        except:
            return
        
        unsafe = 0
        num_path_checks = 12
        path_length = len(path)
        significand = path_length // num_path_checks

        for idx in range(num_path_checks):
            path_point = path[idx * significand]

            for region_idx, region in enumerate(regions):
                if region.contains(path_point):
                    break
            else:
                region_idx = None

            if region_idx is not None and region_idx in obstacles_idx:
                unsafe += 1

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
        
        i = 0
        nonnavigable, unsafe = [], []
        while i < self.configs_to_consider:
            start, goal, obstacles = self._generate_points()
            config_cost = self._config_cost(start, goal, obstacles, regions)
            if config_cost:
                nonnavigable.append(config_cost[0])
                unsafe.append(config_cost[1])
                i += 1
        
        unsafe = sum(unsafe)
        efficiency = len(regions)
        nonnavigable_mu = mean(nonnavigable)
        nonnavigable_var = variance(nonnavigable)
        
        criterion = np.array([unsafe, efficiency, nonnavigable_mu, nonnavigable_var])
        problem_cost = np.sum(criterion * weights)
        return problem_cost

    # Minimizes cost function to generate the optimal lines
    def _generate_optimal_lines(self):
        lb, ub = -1.25, 1.25
        optim_val, optim_lines = inf, None
        start = time.time()
        for num in range(2, 7):
            print(f'Generating language with {num} lines...')
            bounds = [(lb, ub) for _ in range(num*4)]
            res = optimize.differential_evolution(self._optimizer, bounds, disp=True,
                                                  maxiter=500, init='sobol')
            
            print(f'Cost: {res.fun}')
            if optim_val > res.fun:
                optim_val = res.fun
                optim_lines = res.x
        
        print(f'Cost: {res.fun}')
        if optim_val > res.fun:
            optim_val = res.fun
            optim_lines = res.x
        
        end = time.time()
        print(f'Elapsed time: {end - start} seconds')
        optim_lines = np.reshape(optim_lines, (-1, 4))
        return optim_lines
    
    # Visualize regions that define the language
    def _visualize(self, regions):
        for idx, region in enumerate(regions):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        plt.savefig(f'ma-cdl/agents/utils/stored_langs/{self.problem_type}+{weights}.png')
    
    # Returns regions that define the language
    def create(self):
        lines = self._generate_optimal_lines()
        lines = [LineString([line[0:2], line[2:4]]) for line in lines]
        regions = self._create_regions(lines)
        self._visualize(regions)
        self._save(regions)
        return regions