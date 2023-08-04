import os
import io
import pickle
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from statistics import mean
from itertools import product
from shapely.geometry import Point, LineString, MultiLineString, Polygon

warnings.filterwarnings('ignore', message='invalid value encountered in intersection')

CORNERS = list(product((1, -1), repeat=2))
BOUNDARIES = [LineString([CORNERS[0], CORNERS[2]]),
              LineString([CORNERS[2], CORNERS[3]]),
              LineString([CORNERS[3], CORNERS[1]]),
              LineString([CORNERS[1], CORNERS[0]])]
SQUARE = Polygon([CORNERS[2], CORNERS[0], CORNERS[1], CORNERS[3]])

class CDL:
    possible_coeffs = None
    def __init__(self, scenario, world):
        self.min_lines = 1
        self.max_lines = 5
        self.action_dim = 3
        self.granularity = 0.2
        self.world = world
        self.scenario = scenario
        self.configs_to_consider = 100
        self.rng = np.random.default_rng(seed=42)
        
        self.agent_radius = world.agents[0].radius
        self.obstacle_radius = world.large_obstacles[0].radius
        CDL.possible_coeffs = self.create_candidate_set_of_coeffs(granularity=self.granularity)
    
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
                
    # Create regions to be uploaded to Wand 
    def _get_image(self, problem_instance, title_name, title_data, regions, reward):
        _, ax = plt.subplots()
        problem_instance = problem_instance.capitalize()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        ax.set_title(f'Problem Instance: {problem_instance}   {title_name.capitalize()}: {title_data}   Reward: {reward:.2f}')

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
    
    # Generate possible set of lines to choose from
    @staticmethod
    def create_candidate_set_of_coeffs(granularity):
        candidate_lines = []
        granularity = int(granularity * 100)
        
        # vertical/horizontal lines
        for i in range(-100 + granularity, 100, granularity):
            i /= 1000
            candidate_lines += [(0.1, 0, i)] # vertical lines
            candidate_lines += [(0, 0.1, i)] # horizontal lines
        
        # diagonal lines
        for i in range(-200 + granularity, 200, granularity):
            i /= 1000
            candidate_lines += [(0.1, 0.1, i)]
            candidate_lines += [(-0.1, 0.1, i)]
        
        return candidate_lines
            
    # Generate lines (Ax + By + C = 0) from the coefficients
    @staticmethod
    def get_lines_from_coeffs(coeffs):
        lines = []
        equations = np.reshape(coeffs, (-1, 3))
        for equation in equations:
            a, b, c = equation
            
            if a == 0:  # Horizontal line
                start, end = (-1, -c/b), (1, -c/b)
            elif b == 0:  # Vertical line
                start, end = (-c/a, -1), (-c/a, 1)
            else:
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
            if not intersection.is_empty and not intersection.geom_type == 'Point':
                coords = np.array(intersection.coords)
                if np.any(np.abs(coords) == 1, axis=1).all():
                    valid_lines.append(intersection)

        return valid_lines    
    
    # Create polygonal regions from lines
    """WARNING: Changing this distance requires that distance in the safe_graph function be changed"""
    @staticmethod
    def create_regions(valid_lines, distance=2e-4):
        lines = MultiLineString(valid_lines).buffer(distance=distance)
        boundary = lines.convex_hull
        polygons = boundary.difference(lines)
        regions = [polygons] if polygons.geom_type == 'Polygon' else list(polygons.geoms)
        return regions 
    
    # Find the region that contains the entity
    @staticmethod
    def localize(entity, language):
        point = Point(entity)
        try:
            region_idx = list(map(lambda region: region.contains(point), language)).index(True)
        except:
            region_idx = None
        return region_idx
    
    # Map coefficients to the closest possible coefficients
    @staticmethod
    def get_mapped_coeffs(coeffs):
        mapped_coeffs = []
        possible_coeffs = CDL.possible_coeffs
        coeffs = np.reshape(coeffs, (-1, 3))
        for coeff in coeffs:
            distances = [np.linalg.norm(coeff - coefficients) for coefficients in possible_coeffs]
            closest_coeffs_index = np.argmin(distances)
            mapped_coeffs.append(CDL.possible_coeffs[closest_coeffs_index])
        
        return mapped_coeffs
                
    # Generate configuration under specified constraint
    def _generate_configuration(self, problem_instance):
        self.scenario.reset_world(self.world, self.rng, problem_instance)
        
        rand_idx = self.rng.choice(len(self.world.agents))
        start = self.world.agents[rand_idx].state.p_pos
        goal = self.world.agents[rand_idx].goal.state.p_pos
        obstacles = [obs.state.p_pos for obs in self.world.large_obstacles]

        return start, goal, obstacles
    
    # Create graph from language excluding regions with obstacles
    @staticmethod
    def get_safe_graph(regions, obstacles):
        graph = nx.Graph()

        obstacle_regions = [idx for idx, region in enumerate(regions) if any(region.intersects(obstacle) for obstacle in obstacles)]
        
        # Add nodes to graph
        for idx, region in enumerate(regions):
            if idx in obstacle_regions:
                continue
            centroid = region.centroid
            graph.add_node(idx, position=(centroid.x, centroid.y))

        for idx, region in enumerate(regions):
            if idx in obstacle_regions:
                continue

            for neighbor_idx, neighbor in enumerate(regions):
                if idx == neighbor_idx or neighbor_idx in obstacle_regions:
                    continue
                
                if region.dwithin(neighbor, 4.0000001e-4):
                    graph.add_edge(idx, neighbor_idx)

        return graph
        
    """
    Calculate cost of a configuration (i.e. start position, goal position, and obstacle positions)
    with respect to the regions based on the amount of unsafe area (flexibility).
    """
    def _config_cost(self, start, goal, obstacles, regions): 
        def euclidean_distance(a, b):
            return regions[a].centroid.distance(regions[b].centroid)
        
        obstacles_with_size = [Point(obs_pos).buffer(self.obstacle_radius) for obs_pos in obstacles]
    
        graph = CDL.get_safe_graph(regions, obstacles_with_size)
        start_region = CDL.localize(start, regions)
        goal_region = CDL.localize(goal, regions)
        path = []
        try:
            path = nx.astar_path(graph, start_region, goal_region, heuristic=euclidean_distance)
            safe_area = [regions[idx].area for idx in path]
            avg_safe_area = mean(safe_area)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            avg_safe_area = -6
            
        return avg_safe_area
    
    """ 
    Calculate cost of a given problem (i.e. all configurations) 
    with respect to the regions and the given constraints: 
        1. Mean of unsafe area
        2. Variance of unsafe_area
    """
    def optimizer(self, regions, problem_instance):  
        instance_cost = -8
        
        if len(regions) > 1:
            safe_area = []
            efficiency = len(regions) * 0.175
            for _ in range(self.configs_to_consider):
                start, goal, obstacles = self._generate_configuration(problem_instance)
                config_cost = self._config_cost(start, goal, obstacles, regions)
                safe_area.append(config_cost)
        
            safe_area_mu = mean(safe_area)
            instance_cost = safe_area_mu - efficiency
        
        return instance_cost
        
    def _generate_optimal_coeffs(self, problem_instance):
        raise NotImplementedError
        
    # Returns regions that defines the language
    def get_language(self, problem_instance):
        approach = self.__class__.__name__
        try:
            language = self._load(approach, 'cheese')
        except FileNotFoundError:
            print(f'No stored {approach} language for {problem_instance} problem instance.')
            print('Generating new language...\n')
            coeffs = self._generate_optimal_coeffs(problem_instance)
            lines = CDL.get_lines_from_coeffs(coeffs)
            valid_lines = CDL.get_valid_lines(lines)
            language = CDL.create_regions(valid_lines)
            self._visualize(approach, problem_instance, language)
            self._save(approach, problem_instance, language)
        
        return language