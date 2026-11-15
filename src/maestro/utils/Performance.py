# Shared container class to represent a performance. Data types are intentionally left ambiguous

class Performance:
    
    def __init__(self):
        self.video  = None
        self.audio  = None
        self.score  = None
        self.reference = None
        
        self.pose  = None
        self.audio_features = None
        self.score_features = None
        
        self.alignment = None
        self.assessment = None
        self.feedback = None
        