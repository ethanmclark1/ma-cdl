import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from rtree import index
from shapely import points
from itertools import product
from statistics import mean, variance
from languages.utils.potential_field import PathPlanner
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

CORNERS = list(product((1, -1), repeat=2))
BOUNDARIES = [LineString([CORNERS[0], CORNERS[2]]),
              LineString([CORNERS[2], CORNERS[3]]),
              LineString([CORNERS[3], CORNERS[1]]),
              LineString([CORNERS[1], CORNERS[0]])]
SQUARE = Polygon([CORNERS[2], CORNERS[0], CORNERS[1], CORNERS[3]])

class CDL:
    def __init__(self, scenario, world):
        self.max_lines = 8
        self.world = world
        self.language = None
        self.scenario = scenario
        self.configs_to_consider = 30
        self.np_random = np.random.default_rng()
        self.weights = np.array([3, 2, 1.75, 3, 2])
        self.planner = PathPlanner(scenario, world)
    
    def _save(self, approach, problem_instance):
        directory = f'ma-cdl/languages/history/{approach}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(self.language, file)
    
    def _load(self, approach, problem_instance):
        directory = f'ma-cdl/languages/history/{approach}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        self.language = language
    
        # Visualize regions that define the language
    def _visualize(self, approach, problem_instance):
        for idx, region in enumerate(self.language):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y,
                     idx, ha='center', va='center')

        directory = 'ma-cdl/language/history'
        filename = f'{approach}-{problem_instance}.png'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_path)
        plt.cla()
        plt.clf()
        plt.close('all')
        
    # Generate lines (Ax + By + C = 0) from the coefficients
    @staticmethod
    def get_lines_from_coeffs(coeffs):
        lines = []
        equations = np.reshape(coeffs, (-1, 3))
        for equation in equations:
            a, b, c = equation
            # Indicates an infinite slope (invalid line)
            if b == 0:
                continue
            
            slope = a / -b
            if abs(slope) >= 1:
                y1 = (-a + c) / -b
                y2 = (a + c) / -b
                start, end = (-1, y1), (1, y2)
            else:
                x1 = (-b + c) / -a
                x2 = (b + c) / -a
                start, end = (x1, -1), (x2, 1)
            lines.append(LineString([start, end]))

        return lines
    
    # Find the intersections between lines and the environment boundary
    @staticmethod
    def get_valid_lines(lines):
        valid_lines = list(BOUNDARIES)

        for line in lines:
            intersection = SQUARE.intersection(line)
            if not intersection.is_empty:
                coords = np.array(intersection.coords)
                if np.any(np.abs(coords) == 1, axis=1).all():
                    valid_lines.append(intersection)

        return valid_lines    
    
    # Create polygonal regions from lines
    @staticmethod
    def create_regions(valid_lines):
        lines = MultiLineString(valid_lines).buffer(distance=1e-12)
        boundary = lines.convex_hull
        polygons = boundary.difference(lines)
        regions = [polygons] if polygons.geom_type == 'Polygon' else list(polygons.geoms)
        return regions 
                
    # Generate configuration under specified constraint
    def _generate_configuration(self, problem_instance):
        instance_num = problem_instance[-1]
        self.scenario.reset_world(self.world, self.np_random, instance_num)
        
        rand_idx = self.np_random.choice(len(self.world.agents))
        start = self.world.agents[rand_idx].state.p_pos
        goal = self.world.agents[rand_idx].goal_a.state.p_pos
        static_obs = [obs.state.p_pos for obs in self.world.obstacles if obs.movable == False]
        # get dynamic obstacle object instead of position because radius is needed too
        dynamic_obs = [obs for obs in self.world.obstacles if obs.movable == True]

        return start, goal, static_obs, dynamic_obs
    
    # Create RTree index to more efficiently search points
    def _create_index(self, obstacle_points, regions):
        rtree = index.Index()
        for region_idx, region in enumerate(regions):
            rtree.insert(region_idx, region.bounds)

        obstacles_idx = set()
        for obs in obstacle_points:
            for region_idx in rtree.intersection((obs.x, obs.y)):
                if regions[region_idx].contains(obs):
                    obstacles_idx.add(region_idx)
        
        return obstacles_idx, rtree
    
    """
    Calculate cost of a configuration (i.e. start, goal, and obstacles)
    with respect to the regions and positions under some positional constraints: 
        1. Unsafe area caused by all obstacles
        2. Unsafe plan caused by non-existent path from start to goal while avoiding unsafe area
    """
    # TODO: Brainstorm what to do with dynamic obstacles in this case
    def _problem_cost(self, start, goal, static_obs, dynamic_obs, regions):       
        # obstacles = np.concatenate((static_obs, dynamic_obs))
        # obstacle_points = points(obstacles)
        
        # all_obs_idx, rtree = self._create_index(obstacle_points, regions)
        # nonnavigable = sum(regions[idx].area for idx in all_obs_idx)
        
        path = self.planner.get_plan(start, goal, static_obs, dynamic_obs)
        if path is None: return 4, 10
        
        static_obs_points = points(static_obs)
        static_obs_idx, _ = self._create_index(static_obs_points, regions)

        unsafe = 0
        num_path_checks = 10
        significand = len(path) // num_path_checks
        
        for i in range(num_path_checks):
            path_point = path[i * significand]
            point = Point(path_point)            
            for region_idx in rtree.intersection((point.x, point.y)):
                if regions[region_idx].contains(point) and region_idx in static_obs_idx:
                    unsafe += 1
                    break

        return nonnavigable, unsafe
    
    """ 
    Calculate cost of a given problem (i.e. all configurations) 
    with respect to the regions and the given positional constraints: 
        1. Mean of unsafe plans
        2. Variance of unsafe plans
        3. Language efficiency
        4. Mean of nonnavigable area
        5. Variance of nonnavigable area
    """
    def _optimizer(self, regions, problem_instance):            
        i = 0
        nonnavigable, unsafe = [], []
        while i < self.configs_to_consider:
            start, goal, static_obs, dynamic_obs = self._generate_configuration(problem_instance)
            problem_cost = self._problem_cost(start, goal, static_obs, dynamic_obs, regions)
            if problem_cost:
                nonnavigable.append(problem_cost[0])
                unsafe.append(problem_cost[1])
                i += 1

        unsafe_mu = mean(unsafe)
        unsafe_var = variance(unsafe)
        efficiency = len(regions)
        nonnavigable_mu = mean(nonnavigable)
        nonnavigable_var = variance(nonnavigable)

        # No regions were created
        if nonnavigable_mu > 3.95:
            instance_cost = 10e3
        else:
            criterion = np.array([unsafe_mu, unsafe_var, efficiency, nonnavigable_mu, nonnavigable_var])
            instance_cost = np.sum(self.weights * criterion)
            
        return instance_cost
        
    def _generate_optimal_coeffs(self, problem_instance):
        raise NotImplementedError
        
    # Returns regions that defines the language
    def get_language(self, problem_instance):
        approach = self.__class__.__name__
        try:
            self._load(approach, problem_instance)
        except FileNotFoundError:
            print(f'No stored {approach} language for {problem_instance} problem instance.')
            print('Generating new language...')
            coeffs = self._generate_optimal_coeffs(problem_instance)
            lines = CDL.get_lines_from_coeffs(coeffs)
            valid_lines = CDL.get_valid_lines(lines)
            self.language = CDL.create_regions(valid_lines)
            self._save(approach, problem_instance)
        
        self._visualize(approach, problem_instance)