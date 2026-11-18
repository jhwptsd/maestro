# Exceptions for unusable user input (new file).
#
# Raised by preprocessing/ (bad video/audio/score files) and alignment/correspondence.py (score and
# recording are different pieces). Both derive from ValueError and InputError, so an app front end
# can catch `InputError` once and show the message to the user.


class InputError(ValueError):
    """A user-submitted file is missing, unreadable, unsupported or empty."""


class MismatchError(InputError):
    """The submitted score and recording do not appear to be the same piece."""


class OMRUnavailableError(RuntimeError):
    """PDF/image scores need Audiveris, which is not installed (an environment problem, not bad input)."""


class PoseUnavailableError(RuntimeError):
    """No pose source is available (VioPose not configured or failed, audio-only upload, no pose file).
    maestro.analyze() catches this and simply skips the visual analysis."""
