# Conversation with the feedback model about one analysis (new file).
#
# After the written feedback (generator.py) the student can ask follow-up questions ("why is my intonation sharp in bar 12?", "what should
# I practise first?"). Each question is sent with the system prompt below, the same digest of numbers that the feedback was written from
# (and the feedback itself) and the earlier turns, so the model answers about this recording and cannot see anything else of it. The turns are
# kept by gui/server.py per analysis. The model's answers are plain text (shown as text only by the page).
import json

from maestro.feedback import generator, ollama, prompts

MAX_QUESTION = 1000                # characters of one question
MAX_TURNS = 40                     # messages kept in a conversation (20 questions and answers); older ones are dropped from what is sent
MAX_TOKENS = 900

CHAT_SYSTEM = """You are an experienced, encouraging violin teacher in a conversation with a student about ONE recording of their playing. \
You are given the measurements of that recording as JSON, and the written feedback you already gave.

Rules:
- Answer questions about this recording ONLY from those measurements. Never invent a bar, time, number or fault. If the measurements do \
not cover the question, say so plainly.
- General violin technique advice (how to practise, how a shift or bow change works) is welcome; make clear when it is general advice and not \
something measured in this recording.
- "severity" runs 0 (trivial) to 1 (severe); "causes" are hypotheses with a confidence from 0 to 1: say "probably" or "may be", never certain.
- Events with skill "reference" compare the student with a professional's recording (value = student minus professional); answer questions about the \
professional recording from them, as another reading to try, never as a mistake.
- Refer to places by bar number. Be concrete and brief (a short paragraph or a few bullets), warm but honest, plain language.
- Plain text only: no JSON, no markdown headings."""


def system_prompt(digest, feedback=None):
    """CHAT_SYSTEM + the measurements + the feedback already given."""
    text = CHAT_SYSTEM + "\n\nMeasurements:\n" + json.dumps(digest, separators=(",", ":"), ensure_ascii=False)
    if feedback:
        text += "\n\nFeedback already given to the student:\n" + feedback.get("text", "")
    return text


def reply(digest, feedback, history, question, model=None):
    """The model's answer to `question` -> {"reply", "model", "warnings", "usage"}. `history`: earlier turns as
    [{"role": "user" | "assistant", "content": text}, ...] (alternating, starting with the user). LLMError if the question is empty or too
    long, or the call fails."""
    question = (question or "").strip()
    if not question:
        raise ollama.LLMError("Type a question first.")
    if len(question) > MAX_QUESTION:
        raise ollama.LLMError(f"Please keep the question under {MAX_QUESTION} characters.")
    recent = history[-MAX_TURNS:]
    while recent and recent[0]["role"] != "user":               # a window must start with the student
        recent = recent[1:]
    answer, used, note = ollama.complete_with_fallback(generator.resolve(model), system_prompt(prompts.compact(digest), feedback),
                                                       [*recent, {"role": "user", "content": question}], MAX_TOKENS)
    return {"reply": answer["text"].strip(), "model": used, "warnings": [note] if note else [], "usage": answer["usage"]}
