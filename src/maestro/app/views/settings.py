# The settings screen (new file): the writing model (download or choose) and what this device can do.
import flet as ft

from maestro.app import theme, widgets as w
from maestro.utils import capabilities, selftest


def model_card(app, state):
    """The writing model: what is in use, what else is installed, and one-tap downloads of small models (they run on this device through Ollama)."""
    if not state["running"] and state.get("device"):
        return w.card(ft.Column([w.section("Written feedback", ft.Icons.AUTO_AWESOME),
                                 w.muted(f"Using {state['model']}. It runs on this device: nothing is sent anywhere and nothing needs installing.", 13)], spacing=10))
    if not state["running"]:
        head = ft.Row([ft.Icon(ft.Icons.ERROR_OUTLINE, color=theme.OK), ft.Text("Ollama is not running", weight=ft.FontWeight.W_600)], spacing=8)
        note = w.muted("The written feedback is made by a small model running on this device through Ollama (ollama.com). Install it and start it, then come back.", 12)
        return w.card(ft.Column([w.section("Written feedback", ft.Icons.AUTO_AWESOME), head, note], spacing=12))
    rows = [w.muted(f"Using {state['model']}." if state["model"] else "No model downloaded yet: choose one below.", 13)]
    installed = {m["id"] for m in state["installed"]}
    for m in state["suggested"]:
        have = m["id"] in installed
        action = (w.pill("In use" if m["id"] == state["model"] else "Use", theme.GOOD if m["id"] == state["model"] else theme.ACCENT,
                         ft.Icons.CHECK if m["id"] == state["model"] else None) if have else w.pill("Download", theme.ACCENT, ft.Icons.CLOUD_DOWNLOAD))
        rows.append(ft.Container(ft.Row([ft.Column([ft.Text(m["id"], weight=ft.FontWeight.W_600), w.muted(m["label"].split("(", 1)[-1].rstrip(")"), 11)], spacing=1, expand=True), action],
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER), padding=12, border_radius=14, bgcolor=ft.Colors.SURFACE_CONTAINER, ink=True,
                                 on_click=(lambda e, mid=m["id"], h=have: app.run(app.use_model if h else app.get_model, mid))))
    return w.card(ft.Column([w.section("Written feedback", ft.Icons.AUTO_AWESOME)] + rows, spacing=10))


def device_card():
    rows = []
    for name, works, how in capabilities.summary():
        rows.append(ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE if works else ft.Icons.REMOVE_CIRCLE_OUTLINE, size=20, color=theme.GOOD if works else ft.Colors.OUTLINE),
                            ft.Column([ft.Text(name, weight=ft.FontWeight.W_600, size=14), w.muted("Works" if works else how, 12)], spacing=1, expand=True)],
                           spacing=12, vertical_alignment=ft.CrossAxisAlignment.START))
    return w.card(ft.Column([w.section("This device", ft.Icons.PHONELINK_SETUP)] + rows, spacing=14))


def check_card(app, results):
    """"Check this device": a button that really tries each part on a tiny input, and what happened (the ground truth, where the list above is the expectation)."""
    run = w.soft_button("Run the check" if results is None else "Run it again", ft.Icons.PLAY_ARROW_ROUNDED, lambda e: app.run(app.device_check))
    rows = []
    for r in results or []:
        rows.append(ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE if r["ok"] else (ft.Icons.CANCEL if r["core"] else ft.Icons.REMOVE_CIRCLE_OUTLINE), size=20,
                                    color=theme.GOOD if r["ok"] else (theme.BAD if r["core"] else ft.Colors.OUTLINE)),
                            ft.Column([ft.Text(r["name"], weight=ft.FontWeight.W_600, size=14), w.muted(r["detail"], 12)], spacing=1, expand=True),
                            w.muted(f"{r['seconds']} s", 11)], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START))
    if results:
        verdict = selftest.core_ok(results)
        rows += [w.pill("The core works here" if verdict else "Something the phone app needs does not work", theme.GOOD if verdict else theme.BAD),
                 ft.ExpansionTile(title=ft.Text("Text to copy into a bug report", size=13),
                                  controls=[ft.Container(ft.Text(selftest.as_text(results), selectable=True, size=11, font_family="monospace"), padding=12)])]
    return w.card(ft.Column([w.section("Check this device", ft.Icons.FACT_CHECK), w.muted("Tries each part on a tiny example, here and now. Share the result if something fails.", 12), run] + rows,
                            spacing=12))


def build(app, state, results=None):
    top = ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda e: app.home()), w.heading("Settings", 22)], spacing=4)
    return ft.Column([w.centered(ft.Column([top, model_card(app, state), device_card(), check_card(app, results)], spacing=16, horizontal_alignment=ft.CrossAxisAlignment.STRETCH))], scroll=ft.ScrollMode.AUTO, expand=True)
