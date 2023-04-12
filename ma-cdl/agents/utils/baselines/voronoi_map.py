import numpy as np
import networkx as nx

from scipy.spatial import Voronoi, ConvexHull, distance, voronoi_plot_2d
from agents.utils.base_aqent import BaseAgent

# Voronoi Maps are inefficient and not general, as they have to be recomputed each configuration
class VoronoiMap(BaseAgent):
    def __init__(self):
        super().__init__()
        self.voronoi = None
        self.graph = nx.Graph()

    def direct(self, start_pos, goal_pos, obstacles):
        critical_points = np.vstack([start_pos, goal_pos, obstacles])
        self.voronoi = Voronoi(critical_points)
        
        start_index = self.voronoi.point_region[0]
        goal_index = self.voronoi.point_region[1]
        
        for ridge_vertices in self.voronoi.ridge_vertices:
            if -1 not in ridge_vertices:  
                v1, v2 = ridge_vertices
                p1, p2 = self.voronoi.vertices[v1], self.voronoi.vertices[v2] 
                self.graph.add_edge(v1, v2, weight=distance.euclidean(p1, p2)) 
                
        try:
            shortest_path = nx.astar_path(self.graph, start_index, goal_index, weight='weight')
        except:
            return None
        
        return shortest_path

    def find_voronoi_region(self, observation):        
        min_distance = float("inf")
        closest_region = -1

        for i, point in enumerate(self.voronoi.points):
            dist = distance.euclidean(observation, point)
            if dist < min_distance:
                min_distance = dist
                closest_region = self.voronoi.point_region[i]

        return closest_region
    
    def get_action(self, observation, goal, directions, env):
        observation = observation[0:2]
        obs_region = self.find_voronoi_region(observation)
        goal_region = directions[-1]
        if obs_region == goal_region:
            target = goal
        else:
            next_region = directions[directions.index(obs_region)+1:][0]
            region_idx = np.where(self.voronoi.point_region == next_region)[0][0]
            target = self.voronoi.points[region_idx]
        
        action = super().get_action(observation, target, env)
        return action