import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from rtree import index
from shapely import points
from itertools import product
from Signal8 import get_problem
from statistics import mean, variance
from languages.utils.rrt_star import RRTStar
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

CORNERS = list(product((1, -1), repeat=2))
BOUNDARIES = [LineString([CORNERS[0], CORNERS[2]]),
              LineString([CORNERS[2], CORNERS[3]]),
              LineString([CORNERS[3], CORNERS[1]]),
              LineString([CORNERS[1], CORNERS[0]])]
SQUARE = Polygon([CORNERS[2], CORNERS[0], CORNERS[1], CORNERS[3]])

""""Base class for Context-Dependent Languages (EA, TD3, and Bandits)"""
class CDL:
    def __init__(self, agent_radius, obstacle_radius):
        self.max_lines = 8
        self.language = None
        self.configs_to_consider = 30
        self.weights = np.array([3, 2, 1.75, 3, 2])
        self.rrt_star = RRTStar(agent_radius, obstacle_radius)
    
    def _save(self, class_name, scenario):
        directory = f'ma-cdl/languages/history/{class_name}'
        filename = f'{scenario}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(self.language, file)
    
    def _load(self, class_name, scenario):
        directory = f'ma-cdl/languages/history/{class_name}'
        filename = f'{scenario}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        self.language = language
    
        # Visualize regions that define the language
    def _visualize(self, class_name, scenario):
        for idx, region in enumerate(self.language):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y,
                     idx, ha='center', va='center')

        directory = 'ma-cdl/language/history'
        filename = f'{class_name}-{scenario}.png'
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
    def _generate_configuration(self, scenario):
        problem = get_problem(scenario)
        # TODO: Figure out how to represent dynamic obstacles
        dynamic_obs = np.empty((0, 2))
        start_constr = problem['start']
        goal_constr = problem['goal']
        start = np.random.uniform(*zip(*start_constr))
        goal = np.random.uniform(*zip(*goal_constr))
        
        static_obstacle_constr = problem['static_obs']
        if self.dynamic_obstacles:
            dynamic_obstacle_constr = problem['dynamic_obs']
            # Generate random points under constraints to simulate movement of dynamic_obs
            for constr in dynamic_obstacle_constr:
                x_points = np.linspace(constr[0][0], constr[0][1], num=25)
                y_points = np.linspace(constr[1][0], constr[1][1], num=25)
                random_indices = np.random.permutation(len(x_points))
                shuffled_x = x_points[random_indices]
                shuffled_y = y_points[random_indices]
                obs = np.column_stack((shuffled_x, shuffled_y))
                dynamic_obs = np.append(dynamic_obs, obs, axis=0)
            
            num_dynamic = self.num_obstacles // 2
            num_static = self.num_obstacles - num_dynamic
            static_obs = np.array([np.random.uniform(*zip(*static_obstacle_constr)) for _ in range(num_static)])
        else:
            static_obs = np.array([np.random.uniform(*zip(*static_obstacle_constr)) for _ in range(self.num_obstacles)])
        
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
        obstacles = np.concatenate((static_obs, dynamic_obs))
        obstacle_points = points(obstacles)
        
        all_obs_idx, rtree = self._create_index(obstacle_points, regions)
        nonnavigable = sum(regions[idx].area for idx in all_obs_idx)
        
        path = self.rrt_star.plan(start, goal, static_obs)
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
    def _optimizer(self, regions, scenario):            
        i = 0
        nonnavigable, unsafe = [], []
        while i < self.configs_to_consider:
            start, goal, static_obs, dynamic_obs = self._generate_configuration(scenario)
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
            scenario_cost = 10e3
        else:
            criterion = np.array([unsafe_mu, unsafe_var, efficiency, nonnavigable_mu, nonnavigable_var])
            scenario_cost = np.sum(self.weights * criterion)
            
        return scenario_cost
        
    def _generate_optimal_coeffs(self, scenario):
        raise NotImplementedError
        
    # Returns regions that defines the language
    def get_language(self, scenario):
        class_name = self.__class__.__name__
        try:
            self._load(class_name, scenario)
        except FileNotFoundError:
            print(f'No stored {class_name} language for {scenario} problem.')
            print('Generating new language...')
            coeffs = self._generate_optimal_coeffs(scenario)
            lines = CDL.get_lines_from_coeffs(coeffs)
            valid_lines = CDL.get_valid_lines(lines)
            self.language = CDL.create_regions(valid_lines)
            self._save(class_name, scenario)
        
        self._visualize(class_name, scenario)