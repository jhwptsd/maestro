# maestro
[WIP] An app that will take in a video of yourself playing violin alongside a score, and will give personalized feedback

## Status
Implemented: audio extraction from video, score -> MusicXML, note/pitch transcription, score-to-audio alignment,
pitch-deviation and timing assessment, piano-roll score SVG. Vision, fusion and LLM feedback are TODO.

Requires `ffmpeg` on PATH (and optionally Audiveris for PDF/image scores).

```bash
pip install -e .[dev]
pytest                                            # or: PYTHONPATH=src python -m unittest discover -s src/tests
python src/scripts/analyze.py video.mp4 score.musicxml out/
```
