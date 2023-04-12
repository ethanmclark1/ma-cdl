import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapely import Point
from agents.utils.base_aqent import BaseAgent
from agents.utils.path_finder.a_star import a_star
from scipy.spatial import Voronoi, voronoi_plot_2d, distance


class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def direct(self, start_pos, goal_pos, obstacles):
        start_idx = self.localize(Point(start_pos))
        goal_idx = self.localize(Point(goal_pos))
        obstacles = [Point(obstacle) for obstacle in obstacles]
        directions = a_star(start_idx, goal_idx, obstacles, self.language)
        return directions