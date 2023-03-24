from agents.utils.search import search
from agents.utils.base_aqent import BaseAgent

class Speaker(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        
    def direct(self, start, goal, obstacles):
        directions = search(start, goal, obstacles, self.langauge, self.name)
        return directions
