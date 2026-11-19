# HTML report: every value returned by the analysis, with a chart and a plain-language explanation
# (new file).
#
# Input: the Performance filled by maestro.analyze() (assessment["intonation"], ["rhythm"],
# ["correspondence"], audio_features, alignment, score). charts.py draws the figures,
# score/visualize.py draws the annotated score. gui/server.py sends `build_report` output to the
# browser; `standalone_page` wraps it into one self-contained .html file (used by scripts/analyze.py).
import json
from html import escape
from urllib.parse import quote

from maestro.gui import charts
from maestro.gui.viewer import build_viewer
from maestro.score.visualize import score_to_svg

CSS = """
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222}
h2{margin-top:2em;border-bottom:2px solid #eee;padding-bottom:.2em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{border:1px solid #ddd;border-left:6px solid #999;border-radius:6px;padding:10px}
.card b{font-size:1.5em;display:block}.card small{color:#555}
.sec{margin:1em 0 2em}.how{background:#f6f8fa;border-radius:6px;padding:.6em 1em;font-size:.95em}
.scroll{overflow-x:auto;border:1px solid #eee}
.wide .viewer,.wide .vlegend,.wide .vsummary{width:min(1250px,calc(100vw - 2em));position:relative;left:50%;transform:translateX(-50%)}
.viewer{display:grid;grid-template-columns:1fr 1fr;gap:10px}.vcol{min-width:0;border:1px solid #e4e4e4;background:white}
.vlegend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.85em;color:#444;margin:.6em 0}
.vlegend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.vsummary{font-size:.9em;color:#444}
.note.hl{filter:drop-shadow(0 0 3px #0a64ff)}
.events{border-collapse:collapse;width:100%;font-size:.9em;margin:.8em 0}.events td,.events th{border-bottom:1px solid #eee;padding:4px 8px;text-align:left;vertical-align:top}
"""

# Hover a note in either panel of the side-by-side viewer to highlight its partner in the other. Scripts
# inside HTML injected with innerHTML do not run, so the page (index.html / standalone_page) loads this once
# and it listens on the document.
VIEWER_JS = """
document.addEventListener("mouseover", function (e) {
  document.querySelectorAll(".note.hl").forEach(function (n) { n.classList.remove("hl"); });
  var t = e.target.closest && e.target.closest(".note[data-pair]");
  if (!t || !t.dataset.pair) return;
  document.querySelectorAll('.note[data-pair="' + t.dataset.pair + '"]').forEach(function (n) { n.classList.add("hl"); });
});
"""


def _card(label, value, color, help_text):
    return (f'<div class="card" style="border-left-color:{color}"><small>{escape(label)}</small>'
            f'<b>{escape(value)}</b><small>{escape(help_text)}</small></div>')


def _section(title, what, chart, how):
    return (f'<div class="sec"><h2>{escape(title)}</h2><p>{what}</p>{chart}'
            f'<div class="how"><b>How to read it:</b> {how}</div></div>')


def _summary(perf):
    i, r, c = (perf.assessment[k] for k in ("intonation", "rhythm", "correspondence"))
    n_notes = len(perf.score.notes)
    cards = []
    if i["mean_abs_cents"] is not None:
        cards.append(_card("Average pitch error", f"{i['mean_abs_cents']:.0f} cents",
                           charts.grade_color(i["mean_abs_cents"], charts.CENTS_GOOD, charts.CENTS_OK),
                           "Average distance from the written pitch. 100 cents = 1 semitone."))
        o = i["tuning_offset_cents"]
        cards.append(_card("Overall tuning", f"{o:+.0f} cents", charts.grade_color(o, charts.CENTS_GOOD, charts.CENTS_OK),
                           "Positive = you play sharp overall, negative = flat (e.g. violin tuned off A=440)."))
    cards.append(_card("Wrong notes", f"{i['n_wrong_notes']} of {len(i['notes'])}",
                       charts.GOOD if i["n_wrong_notes"] == 0 else charts.OK if i["n_wrong_notes"] <= 0.1 * max(len(i["notes"]), 1) else charts.BAD,
                       "Notes more than a semitone from the score (left out of the pitch averages)."))
    if r["tempo_ratio"] is not None:
        cards.append(_card("Tempo", f"{r['tempo_ratio']:.0%} of written", charts.BLUE,
                           "Your overall speed vs the score's tempo marking. Not an error by itself."))
        cards.append(_card("Average timing error", f"{r['mean_abs_error'] * 1000:.0f} ms",
                           charts.grade_color(r["mean_abs_error"] * 1000, charts.MS_GOOD, charts.MS_OK),
                           "How far notes land from where an even tempo would put them."))
        cards.append(_card("Timing unevenness", f"{r['std_error'] * 1000:.0f} ms",
                           charts.grade_color(r["std_error"] * 1000, charts.MS_GOOD, charts.MS_OK),
                           "Spread of the timing errors; low = steady pulse."))
    sec = perf.assessment.get("section")
    if sec and sec["first_measure"] is not None:
        where = (f"bar {sec['first_measure']}" if sec["first_measure"] == sec["last_measure"]
                 else f"bars {sec['first_measure']}–{sec['last_measure']}")
        cards.append(_card("Section played", where, charts.BLUE,
                           f"The part of the score your recording covers ({sec['coverage']:.0%} of the piece). "
                           "Only this part is assessed."))
    m = min(c["score_match"], c["perf_match"])
    cards.append(_card("Score match", f"{m:.0%}", charts.GOOD if m >= 0.6 else charts.BAD,
                       f"How well the recording fits the score ({n_notes} written notes)."))
    return f'<h2>Summary</h2><div class="cards">{"".join(cards)}</div>'


def _pct(value):
    return "n/a" if value is None else f"{value:.0%}"


def _movement_cards(a):
    """Summary cards for the visual analysis (each only if its result exists)."""
    BL = charts.BLUE
    grade = lambda v, good, ok: charts.BAD if v is None else charts.grade_color(v, good, ok)
    cards = []
    bowing, posture, pitch = (a.get(k) for k in ("bowing", "posture", "visual_pitch"))
    if bowing and bowing["available"]:
        cards += [
            _card("Bow strokes", str(bowing["n_strokes"]), BL, f"About {bowing['strokes_per_second']:.1f} strokes per second."),
            _card("Bow speed evenness", _pct(bowing["stroke_speed_cv"]), grade(bowing["stroke_speed_cv"], 0.2, 0.4),
                  "How much bow speed varies from stroke to stroke (spread / average). Lower = more even."),
            _card("Smoothness within strokes", _pct(bowing["speed_cv_within_stroke"]),
                  grade(bowing["speed_cv_within_stroke"], 0.25, 0.45), "Speed variation inside a stroke. Lower = steadier bow.")]
        if bowing["direction_balance"] is not None:
            cards.append(_card("Down/up balance", f"{bowing['direction_balance']:.2f}",
                               grade(1 - bowing["direction_balance"], 0.15, 0.35),
                               "Shorter stroke direction / longer one; 1.00 = both directions use equal bow."))
    if posture and posture["available"]:
        if posture["shoulder_sway"] is not None:
            cards.append(_card("Torso sway", f"{posture['shoulder_sway']:.2f}", grade(posture["shoulder_sway"], 0.1, 0.25),
                               "How much the shoulders move while playing, in shoulder widths. Lower = steadier."))
        if posture["shoulder_tilt_mean_deg"] is not None:
            cards.append(_card("Shoulder tilt", f"{posture['shoulder_tilt_mean_deg']:+.1f}°",
                               grade(posture["shoulder_tilt_mean_deg"], 5, 10), "Average slope of the shoulder line (0° = level)."))
    if pitch and pitch["available"]:
        cards.append(_card("Left-hand travel", f"{pitch['range_semitones']:.1f} semitones", BL,
                           "How far the left hand moved along the fingerboard, from the video alone (approximate)."))
    return cards


def _movement_sections(perf):
    """Explained charts for the visual analysis, or a notice saying why it is missing."""
    a = perf.assessment
    status = a.get("visual_status", {"available": False, "reason": ""})
    if not status["available"]:
        return ['<div class="sec"><h2>Movement analysis</h2><p>Not available: ' + escape(status["reason"]) + '</p>'
                '<div class="how">The movement analysis needs VioPose, which the project setup installs (<code>setup.ps1</code> / '
                "<code>setup.sh</code>, or <code>python src/scripts/setup_viopose.py</code>). Pitch and timing results above do "
                "not depend on it.</div></div>"]
    motion, parts = perf.motion_features, []
    bowing, pitch = (a.get(k) for k in ("bowing", "visual_pitch"))
    if bowing and bowing["available"]:
        parts.append(_section(
            "Bow movement",
            "The line is how far your right hand travelled along the bow over time. Every change of colour is a bow change.",
            charts.bow_chart(motion),
            "Smooth, regular zigzags mean a steady bow. Long flat stretches are pauses; sudden jumps are tracking glitches "
            "or very fast bow changes."))
        parts.append(_section(
            "Bow strokes",
            "One bar per stroke: how far the hand travelled. Colour shows the direction.",
            charts.stroke_chart(bowing),
            "Long notes use more bow, so lengths differ; judge consistency from the <i>speed evenness</i> card. "
            "Without bow markers the two colours are simply the two directions."))
    parts.append(_section(
        "Arm angles",
        "The angle at each elbow over time (180° = straight arm).",
        charts.arm_chart(motion),
        "The bowing (right) elbow opens and closes with each stroke; the fingering (left) elbow should stay fairly "
        "steady and shifts when you change position. Large unexplained swings suggest tension or tracking noise."))
    if pitch:
        parts.append(_section(
            "Left-hand position as pitch (video only)",
            "The left hand slides toward your body as you play higher on the string. Using the violin's string-length law, "
            "the hand's distance gives how many semitones above the open string the finger is.",
            charts.visual_pitch_chart(pitch),
            "Read it as <i>movement</i>: how far and how fast you shift. It cannot see which string you are on "
            "(on the A string 0 = A4, +7 = E5), and the scale assumes a full-size violin and that your furthest reach is first "
            "position, so absolute pitch can be off by a few semitones."))
    return parts


def build_report(perf):
    """HTML fragment (summary cards + explained charts) for a Performance returned by analyze()."""
    i, r, c = (perf.assessment[k] for k in ("intonation", "rhythm", "correspondence"))
    notes = perf.audio_features["notes"]
    parts = [_summary(perf)]
    cards = _movement_cards(perf.assessment)
    if cards:
        parts.append(f'<h2>Movement summary</h2><div class="cards">{"".join(cards)}</div>')
    parts.append(_section(
        "Does the recording match the score?",
        "Before judging anything we check that you uploaded the right score. Your recording is aligned to "
        "the score note by note and we count the matches. You do not have to play the whole piece: we find the "
        "part of the score you played (see <i>Section played</i>), wherever it starts.",
        charts.match_chart(c),
        "Both bars should pass the dashed line. Mistakes lower the bars a little; a different piece "
        "lowers them a lot. (A recording that stops early lowers the first bar.)"))
    parts.append('<div class="wide">' + _section(
        "Score vs what was played",
        "Left: the score as the app read it from your file. Right: the same score engraved from what was "
        "detected in your recording, note for note on the same layout.",
        build_viewer(perf),
        "Notes are green when in tune, orange or red as they drift, and show the cents sharp (+) or flat (−) "
        "above the staff. A <b>wrong note</b> is drawn on the line you actually played; a dashed outline "
        "with <b>?</b> was written but not heard; purple notes were heard but are not in the score. "
        "Hover a note to find its partner.") + "</div>")
    parts.append(_section(
        "Your performance on the score",
        "Each blue bar is a written note (left to right = time, up = higher pitch). The coloured line on it "
        "marks the pitch you actually played.",
        f'<div class="scroll">{score_to_svg(perf.score, i)}</div>',
        "A line above its bar means sharp, below means flat (one row = one semitone). "
        f"Green = within {charts.CENTS_GOOD} cents, orange = within {charts.CENTS_OK}, red = further off."))
    parts.append(_section(
        "Pitch over time",
        "The grey line is the pitch tracked from your audio. The blue bars show what the score asks for, placed "
        "at the moment you played each note.",
        charts.pitch_trace_chart(perf.audio_features["frames"], notes, perf.score_features, perf.alignment),
        "Where the grey line sits on a blue bar, the note is in tune. Slides, wobble (vibrato) and "
        "scooping into notes show up as curves."))
    parts.append(_section(
        "Intonation: pitch error per note",
        "Each bar is one note: how many cents it was off the written pitch (100 cents = a semitone). "
        "Grey bars are wrong notes, which are not counted as tuning errors.",
        charts.intonation_chart(i),
        "Bars inside the green band are in tune (±%d cents). Up is sharp, down is flat. The dashed blue "
        "line is your average tuning: if most bars lean the same way, retune or check your finger placement."
        % charts.CENTS_GOOD))
    parts.append(_section(
        "Tempo: how fast you played",
        "Each dot is a note: when the score says it should start (at the written tempo) vs when you played it.",
        charts.tempo_chart(perf.score_features, notes, r),
        "The dashed grey line is playing exactly as written. The orange line is your tempo: below it "
        "means slower. Dots away from the orange line are notes that were early or late. "
        "Playing at a different tempo is not penalised."))
    parts.append(_section(
        "Rhythm: timing error per note",
        "After removing your overall tempo and any gradual speeding up or slowing down (judged against the notes "
        "around each one), each bar shows how early (down) or late (up) the note was, in milliseconds.",
        charts.timing_chart(r),
        "Bars inside the green band (±%d ms) are on time. Runs of bars in one direction mean you rushed or "
        "dragged a passage; single tall bars are individual late or early entries." % charts.MS_GOOD))
    return "".join(parts + _assessment_sections(perf) + _movement_sections(perf))


def _assessment_sections(perf):
    """Event-based assessment: scores, when/where things went wrong, why, and the downloadable JSON report."""
    report = perf.assessment.get("synthesis")
    if not report:
        return []
    events = report["events"]
    rows = "".join(
        f"<tr><td>{e['start']:.1f}s</td><td>{'' if e['measure'] is None else e['measure']}</td><td>{escape(e['module'])}</td>"
        f"<td>{escape(e['message'])}</td><td>{escape(e['causes'][0]['explanation']) if e['causes'] else ''}"
        f"{' (' + format(e['causes'][0]['confidence'], '.0%') + ' sure)' if e['causes'] else ''}</td></tr>"
        for e in sorted(events, key=lambda e: -e["severity"])[:15])
    table = (f'<table class="events"><tr><th>time</th><th>bar</th><th>skill</th><th>what</th><th>probable cause</th></tr>{rows}</table>'
             if rows else "<p>No problems were found.</p>")
    links = "".join(f"<li>{escape(l['hypothesis'])} ({l['start']:.1f}s)</li>" for l in report["links"][:6])
    payload = quote(json.dumps(report))
    download = (f'<p><a download="maestro-assessment.json" href="data:application/json;charset=utf-8,{payload}">'
                "Download the full assessment as JSON</a></p>")
    unavailable = "".join(f"<li><b>{escape(m)}</b>: {escape(str(r))}</li>" for m, r in report["unavailable"].items())
    return [
        _section("Assessment: score per skill",
                 "Every skill is judged on its own from the aligned audio, video and score. 100 means nothing to correct; "
                 "each problem found takes off points in proportion to how severe it is.",
                 charts.module_scores_chart(report["scores"]),
                 "Green is good, red needs work. Skills that could not be assessed are listed below the chart."
                 + (f"<ul>{unavailable}</ul>" if unavailable else "")),
        _section("Assessment: when things went wrong",
                 "Each block is one problem found in the recording, on the lane of the skill it belongs to. "
                 "Hover a block to read what it is and the most likely cause.",
                 charts.event_timeline_chart(events),
                 "Blocks of different lanes at the same moment point to a shared cause (see the list below)."
                 + (f"<ul>{links}</ul>" if links else "")),
        _section("Assessment: bars to practise",
                 "The same problems placed on the score's bars, one row per skill.",
                 charts.measure_heatmap(events),
                 "Dark red cells are the worst moment of that skill in that bar: start practising there."),
        _section("Assessment: why it went wrong",
                 "For every problem the analysis names the most likely cause from the measurements (tuning offset, "
                 "leaps, bow speed, posture ...). The bars add up severity x confidence per cause.",
                 charts.cause_chart(report["causes"]),
                 "A tall bar is a habit that explains many problems: fixing it improves several bars at once. Causes "
                 "are informed guesses with the evidence attached, not certainties.")
        + table + download]


def standalone_page(fragment, title="Maestro report"):
    """Wrap a report fragment into a complete HTML document."""
    return (f'<!doctype html><meta charset="utf-8"><title>{escape(title)}</title>'
            f"<style>{CSS}</style><h1>{escape(title)}</h1>{fragment}<script>{VIEWER_JS}</script>")
