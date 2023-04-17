class Speaker():
    def __init__(self):
        a=3
        
    def direct(self, start_pos, goal_pos, obstacles, gather_directions):
        directions = gather_directions(start_pos, goal_pos, obstacles)
        return directions