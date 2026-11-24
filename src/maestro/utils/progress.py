# Progress reporting for analyze() (new file).
#
# analyze() calls `progress(stage_name)` when each stage starts and `progress("done")` at the end (it knows nothing about bars).
# `Tracker` turns those calls into numbers for whoever shows them (gui/server.py for the progress bar of the start page,
# scripts/analyze.py for the terminal): the fraction of the whole job at which the stage starts and ends (stage weights, estimated
# from typical runs: the body-movement stage "pose" is the longest because VioPose runs for minutes, while the audio stages run
# alongside it) and a label for the user. A stage's own progress is not known, so the consumer eases the bar from the start towards
# the end of the stage while it waits. `Tracker.stage("report")` is for the caller to announce the last step (building the page).
import time

STAGES = (                       # (name, weight, label)
    ("video", 0.10, "Converting the video"),
    ("audio", 0.03, "Extracting the audio"),
    ("score", 0.02, "Reading the score"),
    ("pitch", 0.20, "Finding the pitch (the body movement is analysed at the same time)"),
    ("notes", 0.05, "Detecting the notes"),
    ("align", 0.04, "Matching your playing to the score"),
    ("pose", 0.35, "Waiting for the body-movement analysis (the slowest step)"),
    ("visual", 0.05, "Measuring bowing and posture"),
    ("neural", 0.03, "Asking the CLAP model about tone and expression"),
    ("assess", 0.06, "Judging intonation, rhythm, tone and expression"),
    ("report", 0.07, "Building the report"),
)


class Tracker:
    """Callable for analyze(progress=...): `tracker(stage)` records the stage and calls `callback(info)` with
    {"stage", "label", "fraction" (where it starts, 0-1), "ceiling" (where it ends), "started" (time.monotonic())}.
    Stages in `skip` (default: "neural", which only runs on request) are left out of the weights; unknown names are ignored."""

    def __init__(self, callback, skip=("neural",)):
        self.callback = callback
        stages = [s for s in STAGES if s[0] not in skip]
        total = sum(w for _, w, _ in stages)
        self.spans, start = {}, 0.0
        for name, weight, label in stages:
            self.spans[name] = (start / total, (start + weight) / total, label)
            start += weight

    def __call__(self, stage):
        if stage == "done":
            self.callback({"stage": "done", "label": "Done", "fraction": 1.0, "ceiling": 1.0, "started": time.monotonic()})
        elif stage in self.spans:
            start, end, label = self.spans[stage]
            self.callback({"stage": stage, "label": label, "fraction": start, "ceiling": end, "started": time.monotonic()})

    stage = __call__
