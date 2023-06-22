import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree

class VoronoiMap:
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        self.voronoi = None
        
    def _build_graph(self, critical_points):
        G = nx.Graph()
        self.voronoi = Voronoi(critical_points)
        for index, vertex in enumerate(self.voronoi.vertices):
            G.add_node(index, pos=vertex)
            
        for ridge_vertices in self.voronoi.ridge_vertices:
            if -1 not in ridge_vertices:
                v1, v2 = ridge_vertices
                G.add_edge(v1, v2, weight=np.linalg.norm(self.voronoi.vertices[v1] - self.voronoi.vertices[v2]))
        
        return G

    # TODO: Is not working correctly
    def direct(self, start, goal, obstacles):
        critical_points = np.vstack([start, goal, obstacles])
        graph = self._build_graph(critical_points)
        
        tree = KDTree(self.voronoi.vertices)
        start_vertex = tree.query(start)[1]
        goal_vertex = tree.query(goal)[1]

        path = nx.astar_path(graph, start_vertex, goal_vertex, weight='weight')
        print(path)
        self.plot(start, goal, obstacles, path)

        return path
    
    # TODO: Is not working correctly
    def plot(self, start, goal, obstacles, directions):
        fig, ax = plt.subplots()
        voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)

        ax.plot(*start, 'go')
        ax.plot(*goal, 'ro')

        ax.plot(obstacles[:, 0], obstacles[:, 1], 'ko')

        if directions is not None:
            path_points = self.voronoi.vertices[directions]
            ax.plot(path_points[:, 0], path_points[:, 1], 'b-')
        
        # label Voronoi regions
        for i, region in enumerate(self.voronoi.regions):
            if not -1 in region:
                polygon = self.voronoi.vertices[region]
                centroid = polygon.mean(axis=0)
                ax.text(*centroid, str(i), color='blue')

        ax.set_aspect('equal')
        plt.show()