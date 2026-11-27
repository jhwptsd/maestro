# The app's logic, without any user-interface code (new file).
#
# The Flet views (app/views/) only show and collect; everything that does work lives here so it can be tested without a screen and reused by any front
# end: choosing the recording and the score, running the analysis with progress, the written feedback and the chat, and the writing model's status.
# Every method that takes time BLOCKS; the views run them off the interface thread (page.run_thread).
# `Controller(storage)` keeps its files in `storage` (on a phone: the app's own storage folder, which Flet gives in FLET_APP_STORAGE_TEMP).
import os
import shutil
import tempfile
from pathlib import Path

from maestro import analyze
from maestro.feedback import chat, generator, ollama, prompts, settings
from maestro.gui.player import player_data
from maestro.gui.report import media_source
from maestro.score.scan import pages_to_score_file
from maestro.utils import capabilities
from maestro.utils.errors import InputError
from maestro.utils.progress import Tracker

MAX_QUESTIONS = 20               # chat questions per report


class Report:
    """One analysed performance: `data` (gui/player.player_data: traces, events, pose, scores), `digest` (what the model sees), `media` (path of the
    recording to play and its kind), and the written `feedback` and `chat` once there are some."""

    def __init__(self, data, digest, media, kind):
        self.data, self.digest, self.media, self.kind = data, digest, media, kind
        self.feedback, self.chat = None, []


class Controller:
    def __init__(self, storage=None):
        self.storage = Path(storage or os.environ.get("FLET_APP_STORAGE_TEMP") or tempfile.mkdtemp(prefix="maestro_app_"))
        self.storage.mkdir(parents=True, exist_ok=True)
        self.video = self.score = None          # paths chosen or made by the camera
        self.report = None
        self._runs = 0

    # ---- choosing what to analyse -------------------------------------------------------------------------------------
    def set_video(self, path):
        self.video = str(path) if path else None

    def set_score(self, path):
        self.score = str(path) if path else None

    def save_take(self, data, suffix):
        """Bytes recorded by the camera or microphone -> a file in storage, set as the recording."""
        path = self.storage / f"take{suffix}"
        path.write_bytes(data)
        self.set_video(path)
        return path

    def save_scan(self, photos):
        """Photos of score pages (bytes) -> one cleaned file (jpg or pdf), set as the score. InputError / ToolMissingError as scan.py."""
        path = pages_to_score_file(photos, self.storage)
        self.set_score(path)
        return path

    @property
    def ready(self):
        return bool(self.video and self.score)

    # ---- analysis -----------------------------------------------------------------------------------------------------
    def analyze(self, on_progress):
        """Analyse the chosen recording against the chosen score -> Report (also self.report). on_progress({"stage", "label", "fraction",
        "ceiling", ...}) is called as the stages go (utils/progress.Tracker). Raises InputError for unusable files (the message is for the user)."""
        if not self.ready:
            raise InputError("Choose a recording and a score first.")
        self._runs += 1
        work = self.storage / f"run{self._runs}"
        shutil.rmtree(work, ignore_errors=True)
        tracker = Tracker(lambda info: info["stage"] != "done" and on_progress(info), skip=("neural", "reference"))     # (analyze's own "done" comes before the report is built)
        perf = analyze(self.video, self.score, work, pitch_backend=capabilities.pitch_backend(), progress=tracker)
        tracker("report")
        media, kind = media_source(perf)
        digest = prompts.build_digest(perf)
        self.report = Report(player_data(perf, str(media), kind, digest=digest), digest, str(media), kind)
        on_progress({"stage": "done", "label": "Done", "fraction": 1.0, "ceiling": 1.0})
        return self.report

    # ---- the writing model ----------------------------------------------------------------------------------------------
    def write_feedback(self):
        """Written feedback for the current report (kept on it) -> its text. ollama.LLMError (readable) if the model is not available."""
        report = self.report
        try:
            report.feedback = generator.feedback_from_digest(report.digest)
        except ollama.LLMError as e:
            if capabilities.detect()["mobile"] and "not running" in str(e):          # a phone has no Ollama to install: say what is true instead
                raise ollama.LLMError("The written feedback needs a writing model, and this phone cannot run one yet. "
                                      "Everything measured is below.") from None
            raise
        return report.feedback

    def ask(self, question):
        """One chat turn about the current report -> the reply text (kept in report.chat). ollama.LLMError if it cannot be answered."""
        report = self.report
        if len(report.chat) >= 2 * MAX_QUESTIONS:
            raise ollama.LLMError("This conversation has got long: start a new one.")
        answer = chat.reply(report.digest, report.feedback, report.chat, question)
        report.chat += [{"role": "user", "content": question.strip()}, {"role": "assistant", "content": answer["reply"]}]
        return answer["reply"]

    def new_conversation(self):
        if self.report:
            self.report.chat = []

    @staticmethod
    def model_state():
        """{running, model, installed, suggested} for the settings screen (ollama.py): what the writing model is and what could be downloaded."""
        try:
            installed, running = ollama.list_models(), True
        except ollama.LLMError:
            installed, running = [], False
        return {"running": running, "model": settings.get_model() or ollama.pick_model(installed), "installed": installed,
                "suggested": list(ollama.SUGGESTED)}

    @staticmethod
    def choose_model(model):
        if model not in [m["id"] for m in ollama.list_models()]:
            raise ollama.LLMError("That model is not downloaded yet.")
        settings.set_model(model)

    @staticmethod
    def download_model(model, on_progress):
        """Download a model into Ollama, calling on_progress(fraction, label); it is chosen when done. ollama.LLMError if it fails."""
        ollama.pull(model, on_progress)
        settings.set_model(model)
