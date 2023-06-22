import numpy as np
import networkx as nx

class GridWorld:
    def __init__(self, agent_radius, goal_radius, obstacle_radius):
        self.n_bins = (10, 10)
        self.state_ranges = ((-1, 1), (-1, 1))
        self.graph = nx.grid_graph(self.n_bins, periodic=False)
        
    # Convert continuous state to discrete state
    def _discretize_state(self, state):
        discretized_state = []
        
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, np.linspace(self.state_ranges[i][0], self.state_ranges[i][1], self.n_bins[i] + 1)) - 1
            discretized_state.append(min(bin_idx, self.n_bins[i] - 1))
        return tuple(discretized_state)

    def direct(self, start, goal, obstacles):
        temp_graph = self.graph.copy()
        
        start = self._discretize_state(start)
        goal = self._discretize_state(goal)
        obstacles = set(self._discretize_state(obstacle) for obstacle in obstacles)
        for obstacle in obstacles:
            if obstacle != start and obstacle != goal:
                temp_graph.remove_node(obstacle)
        
        directions = nx.astar_path(temp_graph, start, goal, weight='weight')
    
        return directions