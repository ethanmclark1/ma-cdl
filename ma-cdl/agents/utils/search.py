import pdb
import time
import copy
import numpy as np

from queue import PriorityQueue

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return (self.position == other.position).all()
    
def astar(listener, goal, obstacles, env):
    open_list, closed_list = PriorityQueue(), []
    actor_size = listener.size
    start_pos = copy.copy(listener.state.p_pos)
    start_node = Node(None, start_pos)
    start_node.g = start_node.h = start_node.f = 0
    goal_size = goal.size
    goal_pos = copy.copy(goal.state.p_pos)
    goal_node = Node(None, goal_pos)
    goal_node.g = goal_node.h = goal_node.f = 0
    
    open_list.put((0, start_node))
    start_time = time.time()
    
    while not open_list.empty():
        # Timed out
        if time.time() - start_time > 5: return
        try:
            current_node = open_list.get()[1]   
        except IndexError:
            return
        
        closed_list.append(current_node)
        # Currrent node reached goal
        if check_collision([current_node.position, actor_size], 
                           [goal_node.position, goal_size]):
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        # Action 0: no-op
        for action in range(1, env.action_space(listener.name).n):
            listener.state.p_pos = copy.copy(current_node.position)
            env.step(action)

            # If agent collided into a landmark, don't add to children
            for obstacle in obstacles:
                collided = check_collision([listener.state.p_pos, listener.size],
                                           [obstacle.state.p_pos, obstacle.size])
                if collided:
                    break
            if collided:
                continue

            node_position = copy.copy(listener.state.p_pos)
            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if closed_list_contains(closed_list, child): continue
            
            child.g = np.linalg.norm(child.position - current_node.position)
            child.h = np.linalg.norm(goal_node.position - child.position)
            child.f = child.g + child.h
            
            if open_list_contains(open_list, child): continue
            try:
                open_list.put((child.f, child))
            except:
                return
                
def check_collision(entity_a, entity_b):
    collision = False
    size_a, size_b = entity_a[1], entity_b[1]
    # Radius of agent + radius of landmark gives min possible dist without a collision
    min_dist = size_a + size_b 
    dist = np.linalg.norm(entity_a[0] - entity_b[0])
    if min_dist >= dist:
        collision = True
    return collision

def closed_list_contains(closed_list, node):
    for closed_node in closed_list:
        if node.__eq__(closed_node):
            return True
    return False

# No need to compare f values, since they'll be equivalent if positions are
def open_list_contains(open_list, node):
    for open_node in open_list.queue:
        if node.__eq__(open_node[1]):
            return True
    return False