class BaseAgent():
    def __init__(self):
        self.language = None
        
    def set_language(self, language):
        self.language = language
    
    def localize(self, pos):
        return list(map(lambda region: region.contains(pos), self.language)).index(True)
    
    