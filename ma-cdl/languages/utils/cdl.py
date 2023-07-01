import os
import io
import pickle
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from rtree import index
from itertools import product
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

class CDL:
    def __init__(self, scenario, world):
        self.min_lines = 2
        self.max_lines = 6
        self.world = world
        self.scenario = scenario
        self.configs_to_consider = 30
        self.np_random = np.random.default_rng()
        self.weights = np.array([3, 2, 1.5, 3, 2])
            
        self.agent_radius = world.agents[0].size
        self.goal_radius = world.goals[0].size
        self.obstacle_radius = world.large_obstacles[0].size
        
        self.planner = RRTStar(
            self.agent_radius, 
            self.goal_radius, 
            self.obstacle_radius
            )
    
    def _save(self, approach, problem_instance, language):
        directory = f'ma-cdl/languages/history/{approach.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(language, file)
    
    def _load(self, approach, problem_instance):
        directory = f'ma-cdl/languages/history/{approach.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            language = pickle.load(f)
        return language
        
    # Create regions and obstacles to be uploaded to Wand 
    def _get_image(self, problem_instance, title_name, title_data, regions, reward):
        _, ax = plt.subplots()
        problem_instance = problem_instance.capitalize()

        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')

        ax.set_title(f'problem_instance: {problem_instance}   {title_name.capitalize()}: {title_data}   Reward: {reward:.2f}')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        plt.close()
        return pil_image
    
    # Visualize regions that define the language
    def _visualize(self, approach, problem_instance, language):
        plt.clf()
        plt.cla()
        
        for idx, region in enumerate(language):
            plt.fill(*region.exterior.xy)
            plt.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')

        directory = f'ma-cdl/languages/history/{approach.lower()}'
        filename = f'{problem_instance}.png'
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
    
    # Create graph for path planning using networkx
    @staticmethod
    def create_graph(language):
        graph = nx.Graph()
        
        for idx, region in enumerate(language):
            centroid = region.centroid
            graph.add_node(idx, position=(centroid.x, centroid.y))

        for idx, region in enumerate(language):
            for neighbor_idx, neighbor in enumerate(language):
                if idx == neighbor_idx:
                    continue

                if region.touches(neighbor):
                    graph.add_edge(idx, neighbor_idx)

        return graph
    
    # Find the region that contains the entity
    @staticmethod
    def localize(entity, language):
        point = Point(entity)
        try:
            region_idx = list(map(lambda region: region.contains(point), language)).index(True)
        except:
            region_idx = None
        return region_idx
                
    # Generate configuration under specified constraint
    def _generate_configuration(self, problem_instance):
        self.scenario.reset_world(self.world, self.np_random, problem_instance)
        
        rand_idx = self.np_random.choice(len(self.world.agents))
        start = self.world.agents[rand_idx].state.p_pos
        goal = self.world.agents[rand_idx].goal.state.p_pos
        obstacles = [obs.state.p_pos for obs in self.world.large_obstacles]

        return start, goal, obstacles
    
    # Create RTree index to more efficiently search points
    def _create_index(self, obstacles, regions):
        rtree = index.Index()
        for region_idx, region in enumerate(regions):
            rtree.insert(region_idx, region.bounds)

        obstacles_idx = set()
        for obstacle in obstacles:
            for region_idx in rtree.intersection(obstacle.bounds):
                if regions[region_idx].intersects(obstacle):
                    obstacles_idx.add(region_idx)
        
        return obstacles_idx, rtree
    
    """
    Calculate cost of a configuration (i.e. start position, goal position, and obstacle positions)
    with respect to the regions:
        1. Unsafe area caused by all obstacles
        2. Unsafe plan caused by non-existent path from start to goal while avoiding unsafe area
    """
    def _config_cost(self, start, goal, obstacles, regions): 
        obstacles_with_size = [Point(obs_pos).buffer(self.obstacle_radius) for obs_pos in obstacles]
        
        obstacles_idx, rtree = self._create_index(obstacles_with_size, regions)
        nonnavigable = sum(regions[idx].area for idx in obstacles_idx)
        
        path = self.planner.get_path(start, goal, obstacles)
        
        num_path_checks = 10
        if path is not None: 
            unsafe = 0
            indices = np.linspace(0, len(path) - 1, num_path_checks, dtype=int)
            
            for i in indices:
                path_point = path[i]
                agent = Point(path_point).buffer(self.agent_radius)            
                for region_idx in rtree.intersection(agent.bounds):
                    if regions[region_idx].intersects(agent) and region_idx in obstacles_idx:
                        unsafe += 1
                        break
        else:
            unsafe = int(num_path_checks * 1.5)
            
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
    def optimizer(self, regions, problem_instance):  
        instance_cost = 2e2
        
        if len(regions) > 1:
            i = 0
            nonnavigable, unsafe = [], []
            while i < self.configs_to_consider:
                start, goal, obstacles = self._generate_configuration(problem_instance)
                config_cost = self._config_cost(start, goal, obstacles, regions)
                if config_cost:
                    nonnavigable.append(config_cost[0])
                    unsafe.append(config_cost[1])
                    i += 1

            unsafe_mu = mean(unsafe)
            unsafe_var = variance(unsafe)
            efficiency = len(regions)
            nonnavigable_mu = mean(nonnavigable)
            nonnavigable_var = variance(nonnavigable)
            
            criterion = np.array([unsafe_mu, unsafe_var, efficiency, nonnavigable_mu, nonnavigable_var])
            instance_cost = np.sum(self.weights * criterion)
            
        return instance_cost
        
    def _generate_optimal_coeffs(self, problem_instance):
        raise NotImplementedError
        
    # Returns regions that defines the language
    def get_language(self, problem_instance):
        approach = self.__class__.__name__
        try:
            language = self._load(approach, problem_instance)
        except FileNotFoundError:
            print(f'No stored {approach} language for {problem_instance} problem instance.')
            print('Generating new language...\n')
            coeffs = self._generate_optimal_coeffs(problem_instance)
            lines = CDL.get_lines_from_coeffs(coeffs)
            valid_lines = CDL.get_valid_lines(lines)
            language = CDL.create_regions(valid_lines)
            self._save(approach, problem_instance, language)
        
        self._visualize(approach, problem_instance, language)
        return language