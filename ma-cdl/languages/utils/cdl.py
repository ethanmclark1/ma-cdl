import os
import warnings
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from math import inf
from itertools import product
from statistics import mean, variance
from languages.utils.rrt_star import RRTStar
from environment.utils.problems import problem_scenarios
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

""""Base file for Context-Dependent Languages (EA & TD3)"""
class CDL:
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        self.language = None
        self.configs_to_consider = 30
        self.agent_radius = agent_radius
        self.obs_radius = obs_radius
        self.num_obstacles = num_obstacles
        corners = list(product((1, -1), repeat=2))
        self.rrt_star = RRTStar(agent_radius, obs_radius)
        self.square = Polygon([corners[0], corners[2], corners[3], corners[1]])
        self.boundaries = [LineString([corners[0], corners[2]]),
                           LineString([corners[2], corners[3]]),
                           LineString([corners[3], corners[1]]),
                           LineString([corners[1], corners[0]])]
        
    def _save(self):
        raise NotImplementedError
    
    def _load(self):
        raise NotImplementedError
        
    # Generate lines from the coefficients
    # Line in standard form: Ax + By + C = 0
    def _get_lines_from_coeffs(self, coeffs):
        lines = []
        equations = np.reshape(coeffs, (-1, 3))
        
        x, y = sp.symbols('x y')
        for equation in equations:  
            eq = sp.Eq(equation[0]*x + equation[1]*y + equation[2], 0)
            y_expr = sp.solve(eq, y)[0]
            slope = y_expr.as_coefficients_dict()[x]
            if abs(slope) >= 1:
                # Find values of y when x = -1, 1
                solution = sp.solve(eq, y, dict=True)
                start = (-1, solution[0][y].subs(x, -1))
                end = (1, solution[0][y].subs(x, 1))
            else:
                # Find values of x when y = -1, 1
                solution = sp.solve(eq, x, dict=True)
                start = (solution[0][x].subs(y, -1), -1)
                end = (solution[0][x].subs(y, 1), 1)            
            lines.append(LineString([start, end]))
        
        return lines   
    
    # Find the intersections between lines and the environment boundary
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
        regions = [polygons] if polygons.geom_type == 'Polygon' else list(polygons.geoms)
        return regions 
    
    # Generate points under specified constraint
    def _generate_points(self, scenario):
        obstacles = []    
        start_constr = problem_scenarios[scenario]['start']
        goal_constr = problem_scenarios[scenario]['goal']
        obs_constr = problem_scenarios[scenario]['obs']
        
        start = np.random.uniform(*zip(*start_constr))
        goal = np.random.uniform(*zip(*goal_constr))
        
        # set state of obstacles
        if isinstance(obs_constr, tuple):
            obstacles = [np.random.uniform(*zip(*obs_constr)) for _ in range(self.num_obstacles)]
        else:
            obstacles = [np.random.uniform(*zip(*obs_constr[0])) for _ in range(self.num_obstacles // 2)]
            obstacles += [np.random.uniform(*zip(*obs_constr[1])) for _ in range(self.num_obstacles // 2)]
        
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
        1. Unsafe plans
        2. Language efficiency
        3. Mean of nonnavigable area
        4. Variance of nonnavigable area
    """
    def _optimizer(self, coeffs, scenario, weights=np.array([12, 2, 25, 25])):
        lines = self._get_lines_from_coeffs(coeffs)
        regions = self._create_regions(lines)
        if len(regions) == 0: return inf
        
        i = 0
        nonnavigable, unsafe = [], []
        while i < self.configs_to_consider:
            start, goal, obstacles = self._generate_points(scenario)
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
        return problem_cost, regions
    
    # Visualize regions that define the language
    def _visualize(self, class_name, scenario):
        for idx, region in enumerate(self.language):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
            
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}-{scenario}.png'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(file_path)
        plt.cla()
        plt.clf()
        plt.close('all')
    
    def get_language(self):
        raise NotImplementedError