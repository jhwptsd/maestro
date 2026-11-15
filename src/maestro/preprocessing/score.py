# Process score into MusicXML file
#
# Uses utils/convert_file_type.py (and score/omr.py when needed) to yield MusicXML, stored in
# Performance.score. Consumed by score/musicxml.py.
from maestro.utils.convert_file_type import convert_to_musicxml
from maestro.score.musicxml import parse_musicxml


def preprocess_score(perf, score_path, out_dir=None):
    """Convert `score_path` to MusicXML and store the parsed Score in `perf.score`."""
    perf.score = parse_musicxml(convert_to_musicxml(score_path, out_dir))
    return perf
