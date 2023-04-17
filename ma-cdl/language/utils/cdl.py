import os
import pickle
import warnings
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from itertools import product
from language.utils.a_star import a_star
from language.utils.rrt_star import RRTStar
from environment.utils.problems import problem_scenarios
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

""""Base file for Context-Dependent Languages (CDL)"""
class CDL:
    def __init__(self, agent_radius, obs_radius, num_obstacles):
        self.language = None
        self.num_obstacles = num_obstacles
        corners = list(product((1, -1), repeat=2))
        self.rrt_star = RRTStar(agent_radius, obs_radius)
        self.square = Polygon([corners[0], corners[2], corners[3], corners[1]])
        self.boundaries = [LineString([corners[0], corners[2]]),
                           LineString([corners[2], corners[3]]),
                           LineString([corners[3], corners[1]]),
                           LineString([corners[1], corners[0]])]
        
    def _save(self, class_name, scenario):
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}-{scenario}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(self.language, file)
    
    def _load(self, class_name, scenario):
        directory = 'ma-cdl/language/history'
        filename = f'{class_name}-{scenario}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        self.language = language
        
    # Generate lines based off the coefficients
    def _get_lines_from_coeffs(self, coeffs, degree=1):
        lines = []
        equations = np.reshape(coeffs, (-1, degree+1))
        
        x, y = sp.symbols('x y')
        for equation in equations:  
            eq = sp.Eq(equation[0]*x + equation[1]*y, 0)
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
    
    # Determine the intersections between lines and the boundary
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
    
    # Gather points under specified constraint
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
    
    def _generate_optimal_coeffs(self, scenario):
        raise NotImplementedError
    
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
    
    # Returns regions that defines the language
    def get_language(self, scenario):
        class_name = self.__class__.__name__
        try:
            self._load(class_name, scenario)
        except:
            print(f'No stored {class_name} language for {scenario} problem.')
            print('Generating new language...')
            coeffs = self._generate_optimal_coeffs(scenario)
            lines = self._get_lines_from_coeffs(coeffs)
            self.language = self._create_regions(lines)
            self._save(class_name, scenario)
        
        self._visualize(class_name, scenario)
    
    """Generate directions"""
    # Find the region that contains the given position
    def localize(self, pos):
        try:
            region_idx = list(map(lambda region: region.contains(pos), self.language)).index(True)
        except:
            region_idx = None
        return region_idx    
    
    def direct(self, start_pos, goal_pos, obstacles):
        start_idx = self.localize(Point(start_pos))
        goal_idx = self.localize(Point(goal_pos))
        obstacles = [Point(obs) for obs in obstacles]
        directions = a_star(start_idx, goal_idx, obstacles, self.language)
        return directions
    
    """Take actions according to directions"""
    # Find point for listener agent to move to next
    def find_target(self, observation, goal, directions):
        obs_region = self.localize(observation)
        goal_region = self.localize(goal)
        if obs_region == goal_region:
            next_region = goal_region
            target = goal
        else:
            label = directions[directions.index(obs_region)+1:][0]
            next_region = self.language[label]
            target = next_region.centroid
        
        return target