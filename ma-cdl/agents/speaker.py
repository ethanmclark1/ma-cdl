

class Speaker:
    def __init__(self):
        self.language = None
        self.dealer_card = None
    
    def set_language(self, language):
        self.language = language
        
    def set_state(self, dealer_card):
        self.dealer_card = dealer_card
