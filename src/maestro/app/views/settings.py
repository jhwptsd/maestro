# The settings screen (new file): the writing model (download or choose) and what this device can do.
import flet as ft

from maestro.app import theme, widgets as w
from maestro.utils import capabilities


def model_card(app, state):
    """The writing model: what is in use, what else is installed, and one-tap downloads of small models (they run on this device through Ollama)."""
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


def build(app, state):
    top = ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda e: app.home()), w.heading("Settings", 22)], spacing=4)
    return ft.Column([w.centered(ft.Column([top, model_card(app, state), device_card()], spacing=16, horizontal_alignment=ft.CrossAxisAlignment.STRETCH))], scroll=ft.ScrollMode.AUTO, expand=True)
