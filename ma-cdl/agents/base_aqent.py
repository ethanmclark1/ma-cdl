class BaseAgent():
    def __init__(self):
        self.language = None
        
    def set_language(self, language):
        self.language = language
    
    def localize(self, pos):
        for region in self.language:
            if region.contains(pos):
                return self.language.index(region)
    
    