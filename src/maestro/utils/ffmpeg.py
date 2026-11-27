# Locate ffmpeg and ffprobe (new file).
#
# Every part of Maestro that decodes or converts media runs ffmpeg/ffprobe (utils/convert_video_type.py,
# preprocessing/audio.py). They are system programs, and a program installed after a terminal was opened (winget on Windows
# edits the PATH for NEW terminals only) is invisible to a running Python process: "The system cannot find the file
# specified" from deep inside subprocess. `tool(name)` finds the program on PATH, else in the places installers put it
# (on Windows also the PATH as currently stored in the registry, which is what a new terminal would see), and otherwise
# raises ToolMissingError saying how to install it.
import os
import shutil
import sys
from pathlib import Path

from maestro.utils.errors import ToolMissingError

INSTALL_HINT = {"win32": "winget install Gyan.FFmpeg (or run setup.ps1)", "darwin": "brew install ffmpeg (or run setup.sh)"}


def _registry_path_dirs():
    """Directories of the Windows PATH as stored in the registry (machine + user): what a freshly opened terminal sees."""
    try:
        import winreg
    except ImportError:
        return []
    dirs = []
    for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(hive, key) as k:
                dirs += os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]).split(os.pathsep)
        except OSError:
            pass
    return [d for d in dirs if d]


def _extra_dirs():
    """Places to look besides the PATH of this process."""
    if sys.platform == "win32":
        local, program_data = os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramData", "C:\\ProgramData")
        winget = Path(local) / "Microsoft" / "WinGet"
        packages = [p / "bin" for p in sorted((winget / "Packages").glob("*FFmpeg*/ffmpeg-*"))] if local else []
        return (_registry_path_dirs() + [str(winget / "Links")] + [str(p) for p in packages]
                + [r"C:\ffmpeg\bin", os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                   os.path.join(program_data, "chocolatey", "bin"), os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "shims")])
    return ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", os.path.expanduser("~/.local/bin")]


_cache = {}                                                  # name -> path; only successes, so installing it later works


def _found(name, extra):
    if name in _cache and os.path.isfile(_cache[name]):
        return _cache[name]
    path = shutil.which(name)
    for directory in () if path else extra:
        for candidate in (name, name + ".exe"):
            full = os.path.join(directory, candidate)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                path = full
                break
        if path:
            break
    if path:
        _cache[name] = path
    return path


def available(name):
    """True if `name` ("ffmpeg" or "ffprobe") can be found (no exception): a phone has neither, and a WAV recording is still read without them."""
    return _found(name, tuple(_extra_dirs())) is not None


def tool(name):
    """Full path of `name` ("ffmpeg" or "ffprobe"); raises ToolMissingError (with the install command) if it is nowhere."""
    path = _found(name, tuple(_extra_dirs()))
    if path:
        return path
    raise ToolMissingError(f"{name} was not found. Maestro needs ffmpeg (which includes ffprobe): install it with "
                           f"{INSTALL_HINT.get(sys.platform, 'your package manager (e.g. sudo apt-get install ffmpeg)')} "
                           "and open a NEW terminal so the updated PATH is picked up, then try again.")
