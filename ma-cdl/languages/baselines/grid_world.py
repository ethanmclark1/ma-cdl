import numpy as np
import networkx as nx

class GridWorld:
    n_bins = (4, 4)
    def __init__(self):
        self.graph = nx.grid_graph(GridWorld.n_bins, periodic=False)
        
    # Convert continuous state to discrete state
    @staticmethod
    def discretize(state, state_ranges=((-1, 1), (-1, 1))):
        discretized_state = []
        
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, np.linspace(state_ranges[i][0], state_ranges[i][1], GridWorld.n_bins[i] + 1)) - 1
            discretized_state.append(min(bin_idx, GridWorld.n_bins[i] - 1))
        return tuple(discretized_state)
    
    # Convert discrete state to continous state for GridWorld
    @staticmethod
    def dequantize(discretized_state, state_ranges=((-1, 1), (-1, 1))):
        continuous_state = []
        
        for i, val in enumerate(discretized_state):
            bin_width = (state_ranges[i][1] - state_ranges[i][0]) / GridWorld.n_bins[i]
            continuous_val = state_ranges[i][0] + (val + 0.5) * bin_width
            continuous_state.append(continuous_val)
        return tuple(continuous_state)

    def direct(self, start, goal, obstacles):
        def euclidean_distance(node1, node2):
            x1, y1 = node1
            x2, y2 = node2
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        graph = self.graph.copy()
        
        start_node = GridWorld.discretize(start)
        goal_node = GridWorld.discretize(goal)
        obstacle_nodes = set(GridWorld.discretize(obstacle) for obstacle in obstacles)
        obstacle_nodes.discard(start_node)
        obstacle_nodes.discard(goal_node)
        graph.remove_nodes_from(obstacle_nodes)

        try:
            directions = nx.astar_path(graph, start_node, goal_node, heuristic=euclidean_distance)
        except nx.exception.NetworkXNoPath:
            directions = None
            
        return directions