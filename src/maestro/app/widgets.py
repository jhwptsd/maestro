# Small building blocks shared by the screens (new file): cards, pills, the gradient button, score rings, bubbles.
import flet as ft

from maestro.app import theme


def muted(text, size=13, **kw):
    return ft.Text(text, size=size, color=ft.Colors.ON_SURFACE_VARIANT, **kw)


def heading(text, size=20, **kw):
    return ft.Text(text, size=size, weight=ft.FontWeight.W_700, **kw)


def card(content, padding=20, **kw):
    """A rounded surface with a hairline border and a soft shadow."""
    return ft.Container(content, padding=padding, border_radius=theme.RADIUS, bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                        border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                        shadow=ft.BoxShadow(blur_radius=24, color=ft.Colors.with_opacity(0.08, "#000000"), offset=ft.Offset(0, 8)), **kw)


def section(title, icon=None, trailing=None):
    """A section heading with an optional icon and a control on the right."""
    left = ft.Row([ft.Icon(icon, size=20, color=theme.ACCENT)] if icon else [], spacing=8)
    left.controls.append(heading(title, 17))
    return ft.Row([left] + ([trailing] if trailing else []), alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                  vertical_alignment=ft.CrossAxisAlignment.CENTER)


def pill(text, color=theme.ACCENT, icon=None):
    """A small rounded tag."""
    row = ft.Row(([ft.Icon(icon, size=14, color=color)] if icon else []) + [ft.Text(text, size=12, weight=ft.FontWeight.W_600, color=color)],
                 spacing=5, tight=True)
    return ft.Container(row, padding=ft.Padding.symmetric(horizontal=10, vertical=4), border_radius=999, bgcolor=ft.Colors.with_opacity(0.14, color))


def primary_button(label, icon, on_click, enabled=True, height=54):
    """The one big call to action: accent gradient, rounded, dimmed and inert when it cannot be used yet."""
    content = ft.Row([ft.Icon(icon, color="#ffffff", size=22), ft.Text(label, size=16, weight=ft.FontWeight.W_700, color="#ffffff")],
                     alignment=ft.MainAxisAlignment.CENTER, spacing=10)
    return ft.Container(content, height=height, border_radius=height / 2, gradient=theme.gradient(), ink=True, opacity=1 if enabled else 0.4,
                        on_click=on_click if enabled else None, alignment=ft.Alignment.CENTER,
                        shadow=ft.BoxShadow(blur_radius=22, color=ft.Colors.with_opacity(0.35, theme.ACCENT), offset=ft.Offset(0, 8)) if enabled else None)


def soft_button(label, icon, on_click, enabled=True):
    """A quiet secondary action."""
    return ft.FilledTonalButton(content=label, icon=icon, on_click=on_click, disabled=not enabled)


def option(icon, title, subtitle, on_click, enabled=True):
    """A large tappable choice (record / choose a file): icon badge, title, one line of explanation."""
    badge = ft.Container(ft.Icon(icon, color="#ffffff", size=24), width=48, height=48, border_radius=14, gradient=theme.gradient(), alignment=ft.Alignment.CENTER)
    body = ft.Column([ft.Text(title, size=15, weight=ft.FontWeight.W_600), muted(subtitle, 12)], spacing=2, expand=True)
    return ft.Container(ft.Row([badge, body, ft.Icon(ft.Icons.CHEVRON_RIGHT, color=ft.Colors.ON_SURFACE_VARIANT)], spacing=14,
                               vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=14, border_radius=16, ink=True, bgcolor=ft.Colors.SURFACE_CONTAINER, opacity=1 if enabled else 0.45,
                        on_click=on_click if enabled else None)


def score_ring(label, value, size=84, stroke=8):
    """A circular gauge: the score in the middle, its skill below; green, amber or red by the score (None: not assessed)."""
    color = theme.score_color(value) if value is not None else ft.Colors.OUTLINE_VARIANT
    ring = ft.Stack([ft.ProgressRing(value=(value or 0) / 100, stroke_width=stroke, color=color, bgcolor=ft.Colors.with_opacity(0.14, color), width=size, height=size,
                                     stroke_cap=ft.StrokeCap.ROUND),
                     ft.Container(ft.Text("-" if value is None else f"{value:.0f}", size=size * 0.3, weight=ft.FontWeight.W_800), width=size, height=size,
                                  alignment=ft.Alignment.CENTER)], width=size, height=size)
    return ft.Column([ring, muted(label.capitalize(), 12, text_align=ft.TextAlign.CENTER)], spacing=6, horizontal_alignment=ft.CrossAxisAlignment.CENTER)


def bubble(text, mine):
    """One chat message: the student's on the right in the accent colour, the teacher's on the left."""
    return ft.Row([ft.Container(ft.Text(text, selectable=True, color="#ffffff" if mine else None), padding=ft.Padding.symmetric(horizontal=14, vertical=10),
                                border_radius=18, bgcolor=theme.ACCENT if mine else ft.Colors.SURFACE_CONTAINER_HIGH, width=None, margin=ft.Margin.only(left=48 if mine else 0, right=0 if mine else 48))],
                  alignment=ft.MainAxisAlignment.END if mine else ft.MainAxisAlignment.START)


def centered(content, max_width=theme.MAX_WIDTH, padding=ft.Padding.symmetric(horizontal=18, vertical=14)):
    """A column of content centred on a wide screen, full width on a phone."""
    return ft.Row([ft.Container(content, max_width=max_width, padding=padding, expand=True)], alignment=ft.MainAxisAlignment.CENTER,
                  vertical_alignment=ft.CrossAxisAlignment.START)


def mounted(control):
    """True when `control` is on a page (Flet raises instead of answering when it is not, e.g. in tests or before the screen is shown)."""
    try:
        return control.page is not None
    except RuntimeError:
        return False


def walk(control):
    """Every control below `control`, itself included (for tests and for finding things on a screen)."""
    yield control
    for attr in ("controls", "content", "title", "leading", "trailing", "actions"):
        child = getattr(control, attr, None)
        for item in (child if isinstance(child, list) else [child] if child is not None else []):
            if isinstance(item, ft.Control):
                yield from walk(item)


def texts(control):
    """All the text shown below `control`, in order."""
    return [c.value for c in walk(control) if isinstance(c, ft.Text) and c.value is not None]
