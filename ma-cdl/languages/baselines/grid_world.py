import numpy as np
import networkx as nx

class GridWorld:
    def __init__(self, state_ranges=((-1, 1), (-1, 1)), n_bins=(10, 10)):
        self.state_ranges = state_ranges
        self.n_bins = n_bins
        self.graph = nx.grid_graph(n_bins, periodic=False)
        
    # Convert continuous state to discrete state
    def _discretize_state(self, state, n_bins):
        discretized_state = []
        
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, np.linspace(self.state_ranges[i][0], self.state_ranges[i][1], self.n_bins[i] + 1)) - 1
            discretized_state.append(min(bin_idx, n_bins[i] - 1))
        return tuple(discretized_state)

    def direct(self, start, goal, obstacles):
        n_bins = (10, 10)
        temp_graph = self.graph.copy()
        
        start = self._discretize_state(start, n_bins)
        goal = self._discretize_state(goal, n_bins)
        obstacles = set(self._discretize_state(obstacle, n_bins) for obstacle in obstacles)
        for obstacle in obstacles:
            if obstacle != start and obstacle != goal:
                temp_graph.remove_node(obstacle)
        
        directions = nx.astar_path(temp_graph, start, goal, weight='weight')
    
        return directions