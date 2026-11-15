# HTML report: every value returned by the analysis, with a chart and a plain-language explanation
# (new file).
#
# Input: the Performance filled by maestro.analyze() (assessment["intonation"], ["rhythm"],
# ["correspondence"], audio_features, alignment, score). charts.py draws the figures,
# score/visualize.py draws the annotated score. gui/server.py sends `build_report` output to the
# browser; `standalone_page` wraps it into one self-contained .html file (used by scripts/analyze.py).
from html import escape

from maestro.gui import charts
from maestro.score.visualize import score_to_svg

CSS = """
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222}
h2{margin-top:2em;border-bottom:2px solid #eee;padding-bottom:.2em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{border:1px solid #ddd;border-left:6px solid #999;border-radius:6px;padding:10px}
.card b{font-size:1.5em;display:block}.card small{color:#555}
.sec{margin:1em 0 2em}.how{background:#f6f8fa;border-radius:6px;padding:.6em 1em;font-size:.95em}
.scroll{overflow-x:auto;border:1px solid #eee}
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


def build_report(perf):
    """HTML fragment (summary cards + explained charts) for a Performance returned by analyze()."""
    i, r, c = (perf.assessment[k] for k in ("intonation", "rhythm", "correspondence"))
    notes = perf.audio_features["notes"]
    parts = [_summary(perf)]
    parts.append(_section(
        "Does the recording match the score?",
        "Before judging anything we check that you uploaded the right score. Your recording is aligned to "
        "the score note by note and we count the matches. You do not have to play the whole piece: we find the "
        "part of the score you played (see <i>Section played</i>), wherever it starts.",
        charts.match_chart(c),
        "Both bars should pass the dashed line. Mistakes lower the bars a little; a different piece "
        "lowers them a lot. (A recording that stops early lowers the first bar.)"))
    parts.append(_section(
        "Your performance on the score",
        "Each blue bar is a written note (left to right = time, up = higher pitch). The coloured line on it "
        "marks the pitch you actually played.",
        f'<div class="scroll">{score_to_svg(perf.score, i)}</div>',
        "A line above its bar means sharp, below means flat (one row = one semitone). "
        "Green = within 15 cents, orange = within 30, red = further off."))
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
        "After removing your overall tempo, each bar shows how early (down) or late (up) the note was, in "
        "milliseconds.",
        charts.timing_chart(r),
        "Bars inside the green band (±%d ms) are on time. Runs of bars in one direction mean you rushed or "
        "dragged a passage; single tall bars are individual late or early entries." % charts.MS_GOOD))
    return "".join(parts)


def standalone_page(fragment, title="Maestro report"):
    """Wrap a report fragment into a complete HTML document."""
    return (f'<!doctype html><meta charset="utf-8"><title>{escape(title)}</title>'
            f"<style>{CSS}</style><h1>{escape(title)}</h1>{fragment}")
