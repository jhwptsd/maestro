# Shared container class to represent a performance. Data types are intentionally left ambiguous
#
# Passed through every stage: preprocessing fills video/audio/score; audio/, vision/, score/ fill
# the *_features/pose fields; alignment/ fills `alignment`; assessment/ and fusion/ fill `assessment`;
# feedback/ fills `feedback`. `reference` holds a processed pro recording (see reference/).

class Performance:
    
    def __init__(self):
        self.video  = None          # path of the common-format video (preprocessing/video.py), or None
        self.audio_source = None    # path of the original upload that audio is extracted from
        self.audio  = None
        self.score  = None
        self.reference = None
        
        self.timings = {}            # seconds per stage of analyze() (the pose stage runs concurrently with the audio stages)
        self.preprocessing_report = {}  # denoising summaries: {"audio": {...}, "score": {...}, "video": bool}
        self.pose  = None            # vision/pose_io.Pose (3D joints per frame), or None when no pose source is available
        self.motion_features = None  # vision/motion_features.extract_motion output (hands, bow strokes, angles)
        self.audio_features = None
        self.score_features = None
        
        self.embeddings = None       # fusion/embed.py: learned per-window descriptions {audio, video, score} -> {t, z, recon_error}
        self.alignment = None
        self.assessment = None
        self.feedback = None
        