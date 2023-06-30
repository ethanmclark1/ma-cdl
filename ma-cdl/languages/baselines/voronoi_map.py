import matplotlib.pyplot as plt

import networkx as nx
from languages.utils.cdl import CDL
from agents.utils.a_star import a_star
from shapely import Point, MultiPoint, Polygon, MultiPolygon, voronoi_polygons

class VoronoiMap:
    regions = None
    
    def __init__(self, show_animation=True):
        self.show_animation = show_animation
        self.bbox = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    
    @staticmethod
    def voronize(position):
        closest_idx = VoronoiMap.tree.query(position, k=1)[1]
        return closest_idx
    
    @staticmethod
    def get_centroid(idx):
        point = VoronoiMap.voronoi_map.points[idx]
        return point
    
    # Create the voronoi map using Shapely
    @staticmethod
    def create_voronoi_map(self, start, goal, obstacles):
        points = MultiPoint([start, goal, *obstacles])
        voronoi_map = voronoi_polygons(points)
        return voronoi_map

    # Clip the voronoi map to the bounding box
    def clip_voronoi(self, voronoi_map):
        clipped_voronoi = []
        for region in voronoi_map.geoms:
            clipped_polygon = self.bbox.intersection(region)
            if clipped_polygon.is_empty:
                continue
            elif clipped_polygon.geom_type == 'Polygon':
                clipped_voronoi.append(clipped_polygon)
            elif clipped_polygon.geom_type == 'MultiPolygon':
                clipped_voronoi.extend(clipped_polygon)
        return MultiPolygon(clipped_voronoi)
    
    def direct(self, start, goal, obstacles):   
        voronoi_map = self.create_voronoi_map(start, goal, obstacles) 
        clipped_voronoi = self.clip_voronoi(voronoi_map) 
        VoronoiMap.regions = [*clipped_voronoi.geoms]
        
        agent_idx = CDL.localize(Point(start), VoronoiMap.regions)
        goal_idx = CDL.localize(Point(goal), VoronoiMap.regions)
        
        obstacles = [Point(obstacle) for obstacle in obstacles]

        try:
            directions = a_star(agent_idx, goal_idx, obstacles, VoronoiMap.regions)
        except TypeError:
            graph = CDL.create_graph(VoronoiMap.regions)
            directions = nx.shortest_path(graph, agent_idx, goal_idx)
                    
        if self.show_animation:
            print(directions)
            self.plot(start, goal, obstacles)

        return directions
    
    def plot(self, start, goal, obstacle):        
        _, ax = plt.subplots()
        
        plt.scatter(start[0], start[1], color='green', marker='s', label='start')
        plt.scatter(goal[0], goal[1], color='red', marker='*', label='goal')
        
        obstacle_x = [p.x for p in obstacle]
        obstacle_y = [p.y for p in obstacle]
        plt.scatter(obstacle_x, obstacle_y, color='black', marker='x', label='obstacle')
        
        for i, region in enumerate(VoronoiMap.regions):
            centroid = region.centroid
            plt.text(centroid.x, centroid.y, str(i), color='black', fontsize=10)
            polygon = plt.Polygon(region.exterior.coords, color='blue', alpha=0.2)
            ax.add_patch(polygon)
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.legend()
        plt.show()