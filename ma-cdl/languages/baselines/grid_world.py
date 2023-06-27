import numpy as np
import networkx as nx

class GridWorld:
    def __init__(self):
        self.n_bins = (10, 10)
        self.graph = nx.grid_graph(self.n_bins, periodic=False)
        
    # Convert continuous state to discrete state
    @staticmethod
    def discretize(state, n_bins=(10, 10), state_ranges=((-1, 1), (-1, 1))):
        discretized_state = []
        
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, np.linspace(state_ranges[i][0], state_ranges[i][1], n_bins[i] + 1)) - 1
            discretized_state.append(min(bin_idx, n_bins[i] - 1))
        return tuple(discretized_state)
    
    # Convert discrete state to continous state for GridWorld
    @staticmethod
    def dequantize(discretized_state, n_bins=(10, 10), state_ranges=((-1, 1), (-1, 1))):
        continuous_state = []
        
        for i, val in enumerate(discretized_state):
            bin_width = (state_ranges[i][1] - state_ranges[i][0]) / n_bins[i]
            continuous_val = state_ranges[i][0] + (val + 0.5) * bin_width
            continuous_state.append(continuous_val)
        return tuple(continuous_state)

    def direct(self, start, goal, obstacles):
        temp_graph = self.graph.copy()
        
        start = GridWorld.discretize(start)
        goal = GridWorld.discretize(goal)
        obstacles = set(GridWorld.discretize(obstacle) for obstacle in obstacles)
        
        for obstacle in obstacles:
            if obstacle != start and obstacle != goal:
                temp_graph.remove_node(obstacle)
        
        directions = nx.astar_path(temp_graph, start, goal, weight='weight')
        return directions