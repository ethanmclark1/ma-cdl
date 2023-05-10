import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from math import inf
from itertools import product
from statistics import mean, variance
from languages.utils.rrt_star import RRTStar
from environment.utils.problems import problem_scenarios
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
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        self.language = None
        self.configs_to_consider = 30
        self.agent_radius = agent_radius
        self.obs_radius = obs_radius
        self.num_obstacles = num_obstacles
        self.weights = np.array([3, 2, 1.75, 3, 2])
        self.rrt_star = RRTStar(agent_radius, obs_radius)
    
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
        
    # Generate lines from the coefficients
    # Line in standard form: Ax + By + C = 0
    @staticmethod
    def get_lines_from_coeffs(coeffs):
        lines = []
        equations = np.reshape(coeffs, (-1, 3))

        for equation in equations:
            a, b, c = equation

            if b == 0:  # Avoid division by zero
                continue

            slope = a / -b
            abs_slope = abs(slope)

            if abs_slope >= 1:
                # Find values of y when x = -1, 1
                y1 = (-a + c) / -b
                y2 = (a + c) / -b
                start, end = (-1, y1), (1, y2)
            else:
                # Find values of x when y = -1, 1
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
    
    # Generate points under specified constraint
    def _generate_points(self, scenario):
        start_constr = problem_scenarios[scenario]['start']
        goal_constr = problem_scenarios[scenario]['goal']
        obs_constr = problem_scenarios[scenario]['obs']
        
        start = np.random.uniform(*zip(*start_constr))
        goal = np.random.uniform(*zip(*goal_constr))
        
        # set state of obstacles
        obstacles = [np.random.uniform(*zip(*obs_constr)) for _ in range(self.num_obstacles)]

        return start, goal, obstacles
    
    """
    Calculate cost of a configuration (i.e. start, goal, and obstacles)
    with respect to the regions and positions under some positional constraints: 
        1. Unsafe area caused by obstacles
        2. Unsafe plan caused by non-existent path from start to goal while avoiding unsafe area
    """
    def _problem_cost(self, start, goal, obstacles, regions):
        obstacle_points = [Point(obs) for obs in obstacles]
        obstacles_idx = set(idx for idx, region in enumerate(regions)
                            for obs in obstacle_points if region.contains(obs))
        nonnavigable = sum(regions[idx].area for idx in obstacles_idx)
        
        path = self.rrt_star.plan(start, goal, obstacles)
        if path is None: return None

        unsafe = 0
        num_path_checks = 15
        path_length = len(path)
        significand = path_length // num_path_checks
        
        for idx in range(num_path_checks):
            path_point = path[idx * significand]
            for region_idx, region in enumerate(regions):
                if region.contains(Point(path_point)) and region_idx in obstacles_idx:
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
            start, goal, obstacles = self._generate_points(scenario)
            problem_cost = self._problem_cost(start, goal, obstacles, regions)
            if problem_cost:
                nonnavigable.append(problem_cost[0])
                unsafe.append(problem_cost[1])
                i += 1

        unsafe_mu = mean(unsafe)
        unsafe_var = variance(unsafe)
        efficiency = len(regions)
        nonnavigable_mu = mean(nonnavigable)
        nonnavigable_var = variance(nonnavigable)

        criterion = np.array([unsafe_mu, unsafe_var, efficiency, nonnavigable_mu, nonnavigable_var])
        scenario_cost = np.sum(self.weights * criterion)
        if scenario_cost == inf:
            scenario_cost = 10e3
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