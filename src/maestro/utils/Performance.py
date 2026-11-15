# Shared container class to represent a performance. Data types are intentionally left ambiguous
#
# Passed through every stage: preprocessing fills video/audio/score; audio/, vision/, score/ fill
# the *_features/pose fields; alignment/ fills `alignment`; assessment/ and fusion/ fill `assessment`;
# feedback/ fills `feedback`. `reference` holds a processed pro recording (see reference/).

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
        