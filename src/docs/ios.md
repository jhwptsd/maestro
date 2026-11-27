# Running Maestro on an iPhone: what blocks it, how to find out, and the on-device model

Written from code and from what could be checked in the build environment. Where something could not be checked it says so ("unverified").

## Ways to find every feature that cannot run on iOS

No single method is enough, so they are layered from cheapest to most certain. The first three run on any computer and are automated and tested.

| # | method | what it finds | how |
|---|---|---|---|
| 1 | **Static audit** of the code | every third-party package and every started program (subprocess, `os.system`, `shutil.which`), per feature, and whether each is needed or optional | `python src/scripts/ios_audit.py` |
| 2 | **Package-index probe** | whether each package has a wheel for an iPhone (pip is asked for the platform tag `ios_13_0_arm64_iphoneos`: pure-Python wheel works, compiled one needs an iOS build, none = the build fails) | `python src/scripts/ios_audit.py --online --flet` (`--flet` adds Flet's own index, which is where `flet build` gets its wheels) |
| 3 | **Phone sandbox** | what the real pipeline tries that a phone cannot do: imports of packages with no iOS build raise ImportError, no program can be started or found, and every attempt is recorded with its file and line, even when the code catches it | `utils/phone_sandbox.py`; `src/tests/test_phone.py` runs the whole analysis and the app's screens inside it, so a hidden dependency fails the test suite |
| 4 | **Build probe** | anything the two checks above missed in packaging: `flet build ipa` does not compile from source and stops with an error naming a missing binary package | run `flet build ipa` (needs macOS, Xcode, the Flutter SDK) in CI |
| 5 | **On-device self-test** | the ground truth: each part is tried for real on a tiny input on the device (numpy, scipy, WAV, YIN, a whole analysis, scan cleaning, ffmpeg, CREPE, VioPose, Audiveris, the writing model) with times and peak memory | the app: Settings > Check this device; `python -m maestro.utils.selftest [--json]` |
| 6 | **Simulator and device in CI** | what Apple restricts and nothing above can see: memory limits (iOS ends an app that uses too much), background time, file sandbox, store rules | run method 5 in the iOS Simulator, then on the oldest iPhone you support, and watch "Peak memory" |

## What the first three say today

Static audit (offline, summarised): the phone path (WAV reading, YIN, notes and alignment, the four assessment modules, the written-feedback client) imports only
numpy and scipy at the top of a file; CREPE, pYIN, Basic Pitch, CLAP and the Demucs separation import torch, onnxruntime, librosa, transformers and others only
inside functions or `try` blocks, so they degrade instead of breaking the core. The programs started are ffmpeg and ffprobe (any format but WAV, video), Audiveris (reading
a photographed score), VioPose (body movement, TensorFlow in its own Python 3.9) and, outside Python, Ollama. The tests enforce this: adding a top-level import of a package
without an iOS build to the phone path fails `test_phone.py`.

Package-index probe, asked of PyPI for the iPhone tag (Python 3.13):

| package | result on PyPI |
|---|---|
| Pillow, cffi | compiled iOS wheels exist |
| librosa | pure-Python wheel, but it depends on numba, which has none: pYIN is out |
| soundfile | pure wrapper, needs the libsndfile program: out |
| numpy, scipy | no iOS wheel on PyPI. **Unverified** for Flet's index (pypi.flet.dev), which could not be reached from the build environment: run `--online --flet` first, because the core needs both |
| torch, torchaudio, onnxruntime, llama-cpp-python, cryptography | none: CREPE, CLAP, Demucs and Python-side LLMs cannot be installed on iOS |

Phone sandbox: the whole analysis of a WAV recording against a MusicXML score runs with PyTorch, ONNX Runtime, librosa, TensorFlow, OpenCV, transformers and the others
blocked and no program startable. It attempts nothing but the lookups for ffmpeg, ffprobe and Audiveris (each answered "not installed"). The app's home screen, analysis and
report also run inside it.

**Features that cannot run on iOS as built:** any recording format but WAV and any video (ffmpeg), body movement (VioPose), reading a photographed or PDF score (Audiveris),
the CREPE pitch network, pYIN and Basic Pitch, the CLAP tone model, Demucs, cutting a reference video (ffmpeg), and the writing model through Ollama. **Open question:** whether scipy
installs (above); if not, the few scipy calls (`resample_poly`, `stft`/`istft`, filters in the denoiser) need numpy replacements.

## The writing model on the phone

What the written feedback needs on a phone is a small language model that runs in the app, writes 150 to 250 words of natural prose from a short list of measurements, and
answers follow-up questions. Candidates, from current write-ups (none was run here):

| engine | what it is | fit |
|---|---|---|
| Apple's Foundation Models framework | the system's own on-device model (Swift API, iOS 26 and Apple Intelligence devices: iPhone 15 Pro and later Pro models, the 16 and 17 families); no download, no app size, no memory to manage; guided generation can constrain output | best first target where available; excludes older iPhones |
| llama.cpp with Metal | runs any GGUF model (Llama 3.2 1B/3B, Qwen, Gemma) on any recent iPhone; the portable choice | covers the devices Foundation Models does not; you ship or download the weights (1 to 2 GB) and own the memory budget |
| MLX Swift | Apple's framework; fastest in one iPhone benchmark (Qwen 3.5 2B at 61 tokens/s) | fast, but iOS support lags macOS and it needs Metal entitlements |
| LiteRT-LM / MediaPipe LLM inference | Google's runtime; ahead for Gemma models in one benchmark | an alternative if Gemma writes the better lessons |

All of them are Swift, Objective-C or C++, so they reach the app through a Flet extension: a Flutter plugin with a Python control (`flet create --template extension`; a Python
package that bundles Python and Flutter code; Dart changes need a `flet build` to try). **Recommended path:** (1) a small extension over Foundation Models; (2) a llama.cpp extension
for the other devices; both behind the same two methods, `status` and `generate`.

**Done on the Python side:** `feedback/ondevice.py` defines that contract (`Engine`, and `CallbackEngine` that drives an async extension call from the worker thread) and
`feedback/ollama.py` already prefers a registered engine ("device:<model>": no server, no download), so the generator, the chat and the settings screen work with it unchanged; tests
cover it with a stand-in engine. **Not done:** the native engine itself, and everything about it is unmeasured.

**Gates for the first spike** (measure on the oldest iPhone you will support, with the real digest and the real prompt): (a) the model loads and answers inside the app's memory
budget (watch Peak memory in the self-test); (b) a 250-word answer arrives in under about 90 seconds; (c) the text follows the prompt (flowing paragraphs, no lists, names the professional
recording when there are comparison events) in a side-by-side with the Ollama answer on the same digest; (d) the app is not ended in the background or when memory is tight.

## Sources
- [Choosing an on-device LLM runtime on Apple Silicon](https://medium.com/@michael.hannecke/choosing-an-on-device-llm-runtime-on-apple-silicon-a-decision-framework-beyond-benchmarks-2449067b8b67)
- [Bring an LLM provider to the Foundation Models framework (WWDC26)](https://developer.apple.com/videos/play/wwdc2026/339/)
- [On-device LLM benchmarks for Apple Silicon](https://github.com/john-rocky/apple-silicon-llm-bench) and [which runtime is fastest on iPhone](https://dev.to/john-rocky/on-device-llm-on-iphone-which-runtime-is-fastest-mlx-vs-llamacpp-vs-litert-lm-vs-coreml-1b42)
- [Local AI on iPhone 2026: what works](https://www.promptquorum.com/power-local-llm/local-ai-on-iphone-guide)
- [Flet: built-in binary Python packages for Android and iOS](https://flet.dev/docs/reference/binary-packages-android-ios/), [packaging for iOS](https://flet.dev/docs/publish/ios/) and [creating an extension](https://flet.dev/docs/extend/user-extensions/)
