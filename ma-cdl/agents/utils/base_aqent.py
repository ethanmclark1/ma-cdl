class BaseAgent():
    def __init__(self):
        self.language = None
        
    def set_language(self, language):
        self.language = language
    
    def localize(self, pos):
        try:
            region_idx = list(map(lambda region: region.contains(pos), self.language)).index(True)
        except:
            region_idx = None
        return region_idx    