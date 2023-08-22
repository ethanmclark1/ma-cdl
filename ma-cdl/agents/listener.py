import itertools
import numpy as np
import networkx as nx

from languages.utils.cdl import CDL

class Listener:
    def __init__(self, agent):
        size = 30
        self.resolution = 2 / size
        self.graph = nx.grid_graph((size, size), periodic=False)
        self.agent = agent

    # Get the target position for the agent to move towards
    def _get_target(self, agent_pos, goal_pos, directions, language):
        # Get agent region from language
        agent_region = CDL.localize(agent_pos, language)

        try:
            target_region = directions[directions.index(agent_region) + 1]
            centroid = language[target_region].centroid
            target_pos = np.array([centroid.x, centroid.y])
        except IndexError:
            target_pos = goal_pos
        except ValueError:
            return None

        return target_pos
    
    # Inflate obstacles by the size of the agent and remove them from the graph
    def _clean_graph(self, graph, obstacle_nodes, agent_node, target_node):
        inflated_obstacle_nodes = set()
        for obstacle_node in obstacle_nodes:
            for dx, dy in itertools.product(range(-self.inflation_radius, self.inflation_radius + 1), repeat=2):
                inflated_node = (obstacle_node[0] + dx, obstacle_node[1] + dy)
                inflated_obstacle_nodes.add(inflated_node)

        inflated_obstacle_nodes.discard(agent_node)
        inflated_obstacle_nodes.discard(target_node)
        graph.remove_nodes_from(inflated_obstacle_nodes)
        return graph

    def get_action(self, observation, directions, approach, language):
        def euclidean_dist(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        agent_pos = observation[:2]
        goal_pos = observation[2:4]
        # Other agents are grouped in as obstacles
        obstacles = observation[4:-2].reshape(-1, 2)

        target_pos = self._get_target(agent_pos, goal_pos, directions, approach, language)
        if target_pos is None:
            return None

        graph = self.graph.copy()
        agent_node = tuple(map(round, ((np.array(agent_pos) + 1) / self.resolution)))
        target_node = tuple(map(round, ((np.array(target_pos) + 1) / self.resolution)))
        obstacle_nodes = set(tuple(map(round, ((np.array(obstacle) + 1) / self.resolution))) for obstacle in obstacles)

        graph = self._clean_graph(graph, obstacle_nodes, agent_node, target_node)
        
        try:
            path = nx.astar_path(graph, agent_node, target_node, heuristic=euclidean_dist)
            
            next_state = np.array(path[1]) * self.resolution - 1
            direction = (next_state - agent_pos) / np.linalg.norm(next_state - agent_pos)
            
            if np.abs(direction[0]) > np.abs(direction[1]):
                if direction[0] < 0:
                    action = 1  # move left
                else:
                    action = 2  # move right
            else:
                if direction[1] < 0:
                    action = 3  # move down
                else:
                    action = 4  # move up
        except IndexError:
            action = 0 # no-op
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            action = None
        
        return action