# The look of the app (new file): colours, the Material 3 theme, and the helpers that colour a score or a problem.
#
# One violet-to-blue accent over Material 3's own surface roles (ft.Colors.SURFACE_CONTAINER ...), so the light and the dark theme follow the device
# without a second set of colours; only the semantic colours (good / middling / bad) are fixed, chosen to read on both.
import flet as ft

ACCENT = "#7c5cff"
ACCENT_2 = "#38b6ff"
GOOD, OK, BAD = "#2fbf8f", "#f0a93a", "#ef6a6a"
RADIUS = 20
MAX_WIDTH = 760                   # content column: comfortable on a phone, not stretched on a desktop


def gradient(angle=None):
    """The accent gradient (hero, primary button, rings)."""
    return ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT, colors=[ACCENT, ACCENT_2])


def score_color(value):
    """A 0-100 score: green from 85, amber from 60, else red (the same cut-offs as the report's chips)."""
    return GOOD if value >= 85 else OK if value >= 60 else BAD


def severity_color(severity):
    """A problem's severity 0-1: minor, moderate, severe."""
    return "#d9b13b" if severity < 0.34 else OK if severity < 0.67 else BAD


def apply(page):
    """Theme, background and spacing of the page: follows the device's light or dark setting."""
    page.title = "Maestro"
    page.padding = 0
    page.spacing = 0
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.theme = ft.Theme(color_scheme_seed=ACCENT, use_material3=True)
    page.dark_theme = ft.Theme(color_scheme_seed=ACCENT, use_material3=True)
    page.bgcolor = ft.Colors.SURFACE
