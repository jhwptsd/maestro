# Process score into MusicXML file
#
# Validates a user-submitted score, uses utils/convert_file_type.py (and score/omr.py when needed) to
# yield MusicXML, and stores the parsed Score in Performance.score. Consumed by score/musicxml.py.
# Rejects files that are not partwise MusicXML after conversion or that contain no notes.
import xml.etree.ElementTree as ET

from maestro.preprocessing.denoise import denoise_score
from maestro.score.musicxml import parse_musicxml
from maestro.utils.convert_file_type import convert_to_musicxml
from maestro.utils.errors import InputError


def preprocess_score(perf, score_path, out_dir=None, denoise=True):
    """Convert `score_path` to MusicXML and store the parsed Score in `perf.score`.

    Raises InputError for missing/unsupported/corrupt files, timewise MusicXML, or empty scores.
    denoise: remove OMR/MIDI artefacts (preprocessing/denoise.denoise_score); counts go to
    perf.preprocessing_report["score"].
    """
    xml_path = convert_to_musicxml(score_path, out_dir)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        raise InputError("The score is not valid MusicXML.")
    if root.tag != "score-partwise":
        raise InputError(f"Unsupported MusicXML flavour <{root.tag}>; export a partwise score.")
    score = parse_musicxml(xml_path)
    if not score.notes:
        raise InputError("The score contains no notes.")
    if denoise:
        score, perf.preprocessing_report["score"] = denoise_score(score)
        if not score.notes:
            raise InputError("After removing artefacts (notes outside the violin range, glitches) the score is empty.")
    perf.score = score
    return perf
