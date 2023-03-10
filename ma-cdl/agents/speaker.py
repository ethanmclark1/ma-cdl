import copy
import numpy as np

from math import inf
from itertools import product
from shapely import points, Polygon

class Speaker:
    def __init__(self):
        self.language = None
    
    def set_language(self, language):
        self.language = language
    
    # Find region containing point
    def _localize(self, pos):
        for region in self.language:
            if region.contains(pos):
                return region
    
    # Get neighboring regions
    def _get_neighbors(self, cur_region):
        neighbors = []
        for region in self.language:
            if not cur_region.equals_exact(region, 0) and cur_region.dwithin(region, 1e-10):
                neighbors.append(region)
        return neighbors
    
    # Find optimal next region
    def _get_next_region(self, cur_region, goal_region, obstacles):
        neighbor_info = {}
        obstacles = points(obstacles)
        neighbors = self._get_neighbors(cur_region)
        dist_to_goal = cur_region.centroid.distance(goal_region.centroid)
        for neighbor in neighbors:
            # Check if neighbor is goal region
            if neighbor.equals(goal_region):
                next_region = neighbor
                break
            else:
                # A* search to find distance to goal barring obstacles
                g = cur_region.centroid.distance(neighbor.centroid)
                h = neighbor.centroid.distance(goal_region.centroid)
                f = g + h
                neighbor_info[neighbor] = NeighborInfo(f, obstacles)
                
        return next_region
        
    # Gather directions based on path and obstacles
    def direct(self, start, goal, obstacles):
        directions = []
        start, goal = points(start, goal)
        cur_region = self._localize(start)
        goal_region = self._localize(goal)
        directions.append(cur_region)
        
        while not cur_region.equals(goal_region):
            next_region = self._get_next_region(cur_region, goal_region, obstacles)
            directions.append(next_region)
            cur_region = next_region
            
        return directions
        
