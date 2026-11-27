# The Flet app: one code base for Windows, macOS, Linux, Android and iOS (and the browser).
#
# `main(page)` is the Flet entry point (src/main.py runs it). controller.py holds the logic without any interface code; theme.py, widgets.py and
# views/ are the look and the screens. See the README ("The app") for what runs on which device.


def main(page):
    from maestro.app.app import main as run          # (imported when the app starts, so the logic and widgets can be used and tested on their own)
    return run(page)
