# Entry point of the Flet app: `flet run` (desktop), `flet run --web`, and `flet build apk|ipa|windows|macos|linux` all start here.
import flet as ft

from maestro.app import main

if __name__ == "__main__":
    ft.run(main)
