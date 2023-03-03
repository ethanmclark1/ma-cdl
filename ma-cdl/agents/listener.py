import numpy as np

from shapely import Point

class Listener():
    def __init__(self):
        self.language = None
    
    def set_language(self, language):
        self.language = language
    
    def get_action(self, obs, directions):
        a=3
        
    
