import numpy as np
import networkx as nx

"""
GridWorld for generating a language
** Not expressive **
"""
class GridWorld:
    def __init__(self, state_ranges=((-1, 1), (-1, 1)), n_bins=(5, 5)):
        self.state_ranges = state_ranges
        self.n_bins = n_bins
        self.graph = nx.grid_graph(n_bins, periodic=False)
        
    # Convert continuous state to discrete state
    def _discretize_state(self, state):
        discretized_state = []
        
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, np.linspace(self.state_ranges[i][0], self.state_ranges[i][1], self.n_bins[i] + 1)) - 1
            discretized_state.append(min(bin_idx, self.n_bins[i] - 1))  # Clip to the highest bin index if necessary
        return tuple(discretized_state)

    def _dequantize_state(self, discretized_state):
        continuous_state = []
        
        for i, val in enumerate(discretized_state):
            bin_width = (self.state_ranges[i][1] - self.state_ranges[i][0]) / self.n_bins[i]
            continuous_val = self.state_ranges[i][0] + (val + 0.5) * bin_width
            continuous_state.append(continuous_val)
        return tuple(continuous_state)

    # Inexpressive
    def direct(self, start_pos, goal_pos, obstacles):
        temp_graph = self.graph.copy()
        
        start = self._discretize_state(start_pos)
        goal = self._discretize_state(goal_pos)
        obstacles = set(self._discretize_state(obstacle) for obstacle in obstacles)
        for obstacle in obstacles:
            temp_graph.remove_node(obstacle)
        
        try:
            shortest_path = nx.astar_path(temp_graph, start, goal, weight='weight')
        except:
            return None
    
        return shortest_path
    
    def find_target(self, observation, goal, directions):
        obs_region = self._discretize_state(observation)
        goal_region = self._discretize_state(goal)
        if obs_region == goal_region:
            next_region = goal_region
            target = goal
        else:
            next_region = directions[directions.index(obs_region)+1:][0]
            target = self._dequantize_state(next_region)
            
        return target