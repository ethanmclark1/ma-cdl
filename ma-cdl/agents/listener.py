from agents.utils.potential_field import PathPlanner

from languages.utils.cdl import CDL
from languages.baselines.grid_world import GridWorld
from languages.baselines.voronoi_map import VoronoiMap
from languages.baselines.direct_path import DirectPath

class Listener:
    def __init__(self, agent_radius, goal_radius, obstacle_radius, max_observable_dist):
        self.languages = ['EA', 'TD3', 'Bandit']
        self.planner = PathPlanner(agent_radius, goal_radius, obstacle_radius, max_observable_dist)
    
    def get_action(self, observation, directions, approach):
        agent_pos = observation[:2]
        obs_pos = observation[2:-2].reshape(-1, 2)
        goal_pos = observation[-2:]
        
        if approach[0] in self.languages:
            agent_region = CDL.localize(agent_pos, approach[1])
        elif approach[0] == 'voronoi_map':
            agent_region = CDL.localize(agent_pos, VoronoiMap.regions)
        elif approach[0] == 'grid_world':            
            agent_region = GridWorld.discretize(agent_pos)
        
        try:
            target_region = directions[directions.index(agent_region) + 1]
            if approach[0] in [*self.languages, 'voronoi_map']:
                target_pos = target_region.centroid
            elif approach[0] == 'grid_world':
                target_pos = GridWorld.dequantize(target_region)
        except UnboundLocalError:
            agent_idx = DirectPath.get_point_index(agent_pos)
            agent_pos = directions[agent_idx]
            target_pos = directions[agent_idx + 1]
        except ValueError:
            target_pos = goal_pos
            
        action = self.planner.get_action(agent_pos, target_pos, obs_pos)
        return action