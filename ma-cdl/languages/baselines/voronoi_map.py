import numpy as np
import networkx as nx

from scipy.spatial import Voronoi, distance

class VoronoiMap:
    def __init__(self):
        self.voronoi = None
        self.graph = nx.Graph()

    # Inefficient and not general, as it has to be recomputed each configuration
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
                
        # Find the shortest path using Dijkstra's algorithm
        try:
            shortest_path = nx.astar_path(self.graph, start_index, goal_index, weight='weight')
        except:
            return None, None
        
        return shortest_path

    def find_voronoi_region(self, observation):
        min_distance = float("inf")
        closest_region = -1

        for i, point in enumerate(self.voronoi.points):
            dist = distance.euclidean(observation, point)
            if dist < min_distance:
                min_distance = dist
                closest_region = i

        return closest_region