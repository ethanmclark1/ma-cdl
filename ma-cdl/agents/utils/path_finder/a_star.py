import numpy as np 

from queue import PriorityQueue

class Node:
    def __init__(self, parent=None, idx=None):
        self.parent = parent
        self.idx = idx
        
        self.g = 0
        self.h = 0
        self.f = 0
    
    def __eq__(self, other):
        return self.idx == other.idx
    
def a_star(start_idx, goal_idx, obstacles, regions):
    if start_idx == goal_idx:
        return [start_idx]
    
    open_list, closed_list = PriorityQueue(), []
    start_node = Node(None, start_idx)
    goal_node = Node(None, goal_idx)   
    
    open_list.put((0, start_node))
    while not open_list.empty():
        cur_node = open_list.get()[1]
        closed_list.append(cur_node)

        open_regions = list(map(lambda x: x[1], open_list.queue))
        
        successors = []
        cur_region = regions[cur_node.idx]
        for next_region in regions:
            if not cur_region.equals_exact(next_region, 0) and cur_region.dwithin(next_region, 2.0005e-12):
                next_idx = regions.index(next_region)
                successor = Node(cur_node, next_idx)
                successors.append(successor)
        
        for successor in successors:
            if any(regions[successor.idx].contains(obstacles)):
                successor.f = 1e3
                    
            if successor.idx == goal_idx:
                path = []
                current = successor
                while current is not None:
                    path.append(current.idx)
                    current = current.parent
                return path[::-1]
            else:
                if successor in closed_list or successor in open_regions:
                    continue

                successor.g = regions[cur_node.idx].centroid.distance(regions[successor.idx].centroid)
                successor.h = regions[successor.idx].centroid.distance(regions[goal_node.idx].centroid)
                if successor.f == 0:
                    successor.f = successor.g + successor.h
                open_list.put((successor.f, successor))