# MiraV2 Roadmap

> **Status:** planning document. Nothing described here is implemented yet.
> **Baseline:** repository analysis of `miraV2` (development machine: Linux, Wayland + GNOME, PipeWire 1.6.8, CMake 4.4.3, GCC, C++17, no version control).
> **Priority legend:** `MUST` = required for the first working assistant · `SHOULD` = required for a robust assistant · `FUTURE` = valuable but must not block earlier phases.

---

## 1. Vision

MiraV2 is intended to become a modular, Linux-native personal voice/computer assistant: the user speaks, and Mira performs real computer operations — launching applications, searching the web, typing text, manipulating files, controlling the system, reading and understanding the screen, and answering with speech.

Long-term target pipeline:

```
User speaks
    ↓
Microphone
    ↓
Speech-to-Text (Whisper)
    ↓
TAMEV                         ← decides the INTENT
    ↓
Intent
    ↓
Intent-aware Tokenizer        ← extracts information FOR that intent
    ↓
Resolver
    ↓
Action / Tool Executor
    ↓
Linux application / browser / files / keyboard / mouse / screen
    ↓
Verification
    ↓
Response
    ↓
TTS
```

Success means Mira understands natural language and *performs useful computer operations*, not merely returns text. Every stage must be independently testable and replaceable, and no stage may be permanently bound to one desktop environment or one external tool.

---

## 2. Current State

A factual snapshot from direct repository inspection. This is the starting point the plan is built on.

### 2.1 What actually compiles today

A single CMake target `miraV2` built from five sources (`CMakeLists.txt`):

| Source | Role | State |
|---|---|---|
| `main.cpp` | orchestration (33 lines) | functional, print-only |
| `speech/whisper.cpp` | Whisper wrapper (42 lines) | thin, functional |
| `audio/microphone.cpp` | PipeWire capture (384 lines) | functional but fragile |
| `input/hotkey.cpp` | evdev hotkey listener | functional on this machine only |
| `cmdmgr/cmdmgr.cpp` | naive first-word splitter (84 lines) | functional, architecturally obsolete |

**Not in the build at all:** `tokenizer/`, `appmgr/`, `screen_reader/`, `models/`, plus a legacy standalone prototype (`live.cpp`) that was present at the root when this analysis began and has since been removed from the project.

### 2.2 Actual runtime flow today

```
Hotkey (/dev/input/event3, hardcoded)
    → Microphone (PipeWire, fixed 3 s window)
    → Whisper (models/ggml-base.en.bin, relative path)
    → CommandManager (action = first word, target = rest)
    → std::cout and exit
```

No intent classification, no JSON configuration, no launching, no input injection, no screen capture, no verification, no TTS.

### 2.3 Component inventory

| Component | State | Evidence |
|---|---|---|
| `Whisper` | Implemented, thin | `speech/whisper.{hpp,cpp}`; `models/ggml-base.en.bin` (148 MB) present |
| `Microphone` | Implemented, fragile | `audio/microphone.cpp`; PipeWire F32 @ 16 kHz mono |
| `Hotkey` | Implemented, dev-machine only | `input/hotkey.cpp`; hardcoded `/dev/input/event3`, chord CTRL+SHIFT+SUPER+SPACE (keycodes 29/42/125/57) |
| `CommandManager` | Implemented, to be retired | `cmdmgr/cmdmgr.cpp` |
| `Tokenizer` | Implemented ~15 % of intended scope | `tokenizer/tokenizer.cpp`; `OPEN_APPLICATION` only |
| `tokenizer/test_tokenizer` | Hand-compiled binary, no build rule | 48 KB binary, not referenced by CMake/ctest |
| `AppManager` | **Empty placeholder** | `appmgr/appmgr.hpp` and `appmgr/appmgr.cpp` are 0 bytes |
| `ScreenReader` | **Empty placeholder** | `screen_reader/SReader.hpp` and `SReader.cpp` are 0 bytes |
| Action executor | **Does not exist** | no `system(`/`popen`/`execv`/`fork(`/`posix_spawn` anywhere in project C++ |
| Application resolver | **Does not exist** | no config directory, no JSON usage anywhere in project C++ |
| TAMEV in C++ | **Does not exist** | ONNX Runtime not referenced by CMake |
| C++ WordPiece tokenizer | **Does not exist** | only `models/tamev-base/tokenizer.json` (WordPiece, 30522, BERT normalizer, lowercase); no `vocab.txt` |
| legacy `live.cpp` prototype | **Removed from the project root during analysis** | it was an orphaned standalone program with its own `main()`, its own S16 PipeWire capture, a hardcoded Bluetooth node (`bluez_input.05:17:6F:5B:35:65`) and a hardcoded `whisper.cpp/models/ggml-base.en.bin` path; an identical copy still exists outside the project at `/home/arc/mira-tools/live.cpp` |
| TAMEV Python side | Working workflows | `models/tamev-base/` (TinyBERT 4L/312D, 14.39 M params, tier "nano"), `models/training/mira-tamev/` fine-tune, `models/tester.py`, `models/runners/tester.py`, `models/training/{train.py,dataset.py,split_data.py}` |
| TTS | **Does not exist** | no `piper`/`espeak-ng`/`espeak`/`spd-say`/`festival`/`flite` installed |
| Input automation | **Does not exist** | no `ydotool`/`wtype`/`xdotool` installed |

### 2.4 ML findings that shape the plan

- The ONNX artifact contract (`models/model_int8.onnx`, 33 MB) is confirmed: inputs `ctx_input_ids` / `ctx_attention_mask` of shape `(B, ctx_len)`, `opt_input_ids` / `opt_attention_mask` of shape `(B, K, opt_len)`, all `int64`; outputs `logits` / `probs` of shape `(B, K)`, `float32`.
- **Provenance:** `models/.cache/huggingface/download/model_int8.onnx.metadata` exists (HF commit hash + etag + timestamp) → the ONNX file was **downloaded from the Hugging Face hub (the released base artifact)**, not exported from the local fine-tune. There is **no export script** in the repository (`torch.onnx.export` / `onnx.quantize` / `quantize_dynamic` / `optimum` → zero matches).
- **Training-data gap:** `train.py` reads `train.jsonl` (56 rows) / `validation.jsonl` (7) / `test.jsonl` (7), produced by `split_data.py` from the 70 hard-coded examples in `dataset.py`. The larger `models/training/datasets/open_application.jsonl` (1165 rows) and `search_web.jsonl` (1016 rows) — **2181 usable examples — are never read by any script.** This is the most likely cause of the ~57 % test accuracy.
- **Input-format inconsistency:** `models/tester.py` builds context as `state + " " + "What should Mira do?"` with 4 options and option padding 32, while `train.py` builds `"Which action should Mira perform?\nUser request: " + text` with 7 options and option padding 64. One standard format must be chosen and shared.
- **Option order is a hard contract:** `train.py` labels are `OPTIONS.index(label)`, so the candidate-list order is baked into the logits indices. Any C++ option list must match it exactly.

### 2.5 Known defects in current code (verified)

- `input/hotkey.cpp`: `read()` is called without checking `keyboard_fd >= 0` → if the device fails to open, both wait loops busy-spin at 100 % CPU. No `EVIOCGRAB`, no polling, no device discovery, no configurable chord.
- `audio/microphone.cpp`: `node_bound` and `audio_samples` are written from the PipeWire thread and read from the main thread without synchronisation (data races); `stop()` only clears a flag and never quits the main loop; the main loop runs on a **detached** thread that outlives the object; the destructor is empty (no stream disconnect, no loop/context destroy, no `pw_deinit`); capture is a fixed 3-second window with no VAD/endpointing; the registry listener binds whichever `Audio/Source` node is seen last.
- `main.cpp`: the Whisper model path is relative to the current working directory.
- `speech/whisper_lib/lib/libwhisper.so.1` has `RUNPATH=/home/arc/mira-tools/whisper.cpp/build/bin` → at runtime the ggml libraries load from a directory **outside the project**, so the build is not self-contained.
- `CMakeLists.txt`: no `CMAKE_BUILD_TYPE`, no warnings, no tests, and `pkg_check_modules(PIPEWIRE REQUIRED)` makes PipeWire a hard configure-time requirement.
- No `.gitignore`, and **no git repository exists** for `miraV2` or its parent directory.

### 2.6 Intent vocabulary inconsistency (blocking issue)

Three incompatible vocabularies exist with no shared type:

| Location | Form | Example |
|---|---|---|
| `tokenizer.cpp` | uppercase string | `"OPEN_APPLICATION"` |
| `cmdmgr.cpp` | lowercase raw verb | `"open"` |
| `train.py` | lowercase snake_case option | `"open_application"` |

There is no `enum class Intent` anywhere in C++. Defining one canonical representation is a prerequisite for connecting TAMEV, the tokenizer, the resolver and the executor.

### 2.7 Environment capabilities relevant to later phases

Present: ONNX Runtime C/C++ API 1.29 (headers `/usr/include/onnxruntime`, CMake config `/usr/lib/cmake/onnxruntime`), `nlohmann-json 3.12.0` (header-only), `libportal 0.11.0` (pkg-config name is `libportal`), `libei-1.0` / `liboeffis-1.0` 1.6.0, X11/XCB/XFixes/XRandR, GTK3/GTK4, `dbus-1` 1.16.2, OpenCV, PipeWire 1.6.8, `xdg-desktop-portal-gnome`, `ffmpeg`, `import` (ImageMagick), `xdg-open`, `gdbus`.

Absent: `ydotool`, `wtype`, `xdotool`, `tesseract`, all TTS engines, `gnome-screenshot`. `grim` is installed but is wlroots-only and **will not work on GNOME Wayland** — the design must not depend on it. The user is in the `input` group, which is why `/dev/input/event*` is readable today.

Python: system `python3` has no `onnxruntime`; `/home/arc/ml_env` provides `torch 2.13.0+cu130`, `transformers 5.14.1`, `onnxruntime 1.30.0` (note the 1.30 vs system 1.29 skew for parity testing).

### 2.8 Gap summary

Currently implemented: capture, transcription, hotkey detection, and a naive text splitter. Missing: **intent classification, resolution, execution, input automation, screen access, verification, TTS, configuration, logging, and tests**. The remainder of this document is a dependency-ordered plan to close that gap without rewriting the parts that work.

## 3. Target Architecture

### 3.1 System layers

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                              MIRA RUNTIME (C++)                               │
│                                                                               │
│  ┌───────┐   ┌──────────┐   ┌────────┐   ┌───────────┐   ┌───────────────┐    │
│  │ INPUT │   │  AUDIO   │   │ SPEECH │   │   TAMEV   │   │   TOKENIZER   │    │
│  │ Hotkey│──▶│Microphone│──▶│ Whisper│──▶│  (ONNX)   │──▶│ intent-aware  │    │
│  │Listen.│   │(PipeWire)│   │        │   │  intent   │   │  extraction   │    │
│  └───────┘   └──────────┘   └────────┘   └───────────┘   └───────┬───────┘    │
│                                                                  │            │
│                                          Intent + TokenResult    │            │
│                                                                  ▼            │
│                                                        ┌───────────────────┐  │
│                                                        │    EXECUTOR /     │  │
│                                                        │    DISPATCHER     │  │
│                                                        └─────────┬─────────┘  │
│              ┌────────────────────┬────────────────────┬─────────┴─────────┐  │
│              ▼                    ▼                    ▼                   ▼  │
│      ┌───────────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────┐ │
│      │   RESOLVER    │   │  APP MANAGER   │   │    INPUT     │   │ SCREEN   │ │
│      │  AppResolver  │──▶│   launcher     │   │  automation  │   │ READER   │ │
│      │ (config/apps) │   │ (LaunchSpec)   │   │ (backends)   │   │(backends)│ │
│      └───────────────┘   └────────────────┘   └──────────────┘   └────┬─────┘ │
│         (no launching)     (no discovery)      (no exec logic)        │       │
│                                                                       ▼       │
│                                                            ┌────────────────┐  │
│                                                            │ VISION / SCREEN│  │
│                                                            │ UNDERSTANDING  │  │
│                                                            │ (OpenCV/OCR)   │  │
│                                                            └───────┬────────┘  │
│                                                                    ▼           │
│                                                          ┌──────────────────┐  │
│                                                          │   VERIFICATION   │  │
│                                                          └────────┬─────────┘  │
│                                                                   ▼            │
│                                                          ┌──────────────────┐  │
│                                                          │   RESPONSE/TTS   │  │
│                                                          └──────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
        ▲                                                        │
        │                    stable interfaces                   │
        └────────────────── platform backends ◀──────────────────┘
                (Portal · Wayland · X11 · GNOME-specific)
```

### 3.2 Layering rule

| Layer | Contains | May depend on |
|---|---|---|
| **core** | `Intent`, intent strings, `ActionRequest`, `Result`, shared text utilities, config paths, logging | nothing |
| **engines** | tamev, tokenizer, resolver, appmgr, executor | core |
| **platform** | input backend, screen-reader backend, vision/OCR backend, TTS backend | core (never on engines) |
| **app** | `main.cpp`, wiring, the voice loop | everything, but nothing depends on it |
| **tests** | unit + integration + parity harnesses | the layer under test |

Platform code is reachable only through abstract interfaces declared in the engine layer and implemented in the platform layer, so supporting a new Linux environment means *adding a backend*, not editing Mira's logic.

### 3.3 Target directory layout

Additive. Existing working files stay where they are until their phase moves them.

```
miraV2/
├── ROADMAP.md              ← this document
├── CMakeLists.txt          ← extended in Phase 0/1 (core lib + app + tests)
├── config/
│   └── apps.json           ← user application configuration (Phase 3)
├── core/                   ← Phase 1
│   ├── intent.hpp/.cpp          canonical enum + string conversion
│   ├── action.hpp               ActionRequest + TokenResult handoff types
│   ├── result.hpp               Result/Status + error codes
│   ├── text_utils.hpp/.cpp      normalize/trim (moved, not rewritten)
│   ├── paths.hpp/.cpp           config/model/log path resolution
│   └── log.hpp/.cpp             minimal levelled logging
├── tamev/                  ← Phase 6
│   ├── bert_tokenizer.hpp/.cpp  WordPiece from tokenizer.json
│   ├── option_catalog.hpp       candidate list in exact training order
│   └── tamev_classifier.hpp/.cpp ONNX Runtime session
├── tokenizer/              ← extend existing files (Phase 2)
│   ├── tokenizer.hpp/.cpp
│   └── test_tokenizer.cpp
├── resolver/               ← Phase 3
│   ├── application_config.hpp/.cpp
│   └── app_resolver.hpp/.cpp
├── appmgr/                 ← Phase 4 (files currently 0 bytes)
│   ├── appmgr.hpp/.cpp
│   └── launch_spec.hpp
├── executor/               ← Phase 5
│   ├── action_dispatcher.hpp/.cpp
│   └── actions/                 one action per intent
├── input/                  ← existing hotkey + Phase 9
│   ├── hotkey.hpp/.cpp          listener (fixed in Phase 14)
│   └── input_backend.hpp + x11_xtest.cpp / portal_eis.cpp / ydotool.cpp
├── screen_reader/          ← Phase 10 (files currently 0 bytes)
│   ├── SReader.hpp              abstract interface + capability probe
│   └── sr_portal.cpp / sr_x11.cpp / sr_wayland.cpp
├── vision/                 ← Phase 11
│   ├── screen_analysis.hpp/.cpp
│   └── ocr_backend.hpp + tesseract_backend.cpp
├── verification/           ← Phase 12
│   └── verifier.hpp/.cpp
├── tts/                    ← Phase 13
│   └── tts_engine.hpp + piper_backend.cpp / espeak_backend.cpp
├── speech/, audio/         ← existing, hardened in Phase 14
├── cmdmgr/                 ← retired in Phase 1 (see 5.9)
└── tests/                  ← added every phase
```

---

## 4. Core Principles

These are binding. Every phase is reviewed against them.

1. **TAMEV decides intent.** The intent classifier runs first and is the only component allowed to choose *what* Mira does.
2. **The tokenizer extracts information after intent is known.** It never guesses intent, and never runs before TAMEV.
3. **The resolver finds resources/entities.** Names → concrete specifications. It does not execute.
4. **The manager/tool performs actions.** It launches/types/captures what it was given. It does not discover.
5. **The executor coordinates.** It maps `Intent + TokenResult` to the right collaborator and returns a `Result`.
6. **Verification checks results.** Separate from execution, and pluggable.
7. **Components communicate through stable interfaces**, not through shared globals or hidden coupling.
8. **Platform-specific code stays behind backends**, selected by capability detection, never by hardcoded desktop assumptions.
9. **No unnecessary ML where deterministic code is sufficient** — string/path/alias work is deterministic.
10. **ML is used where it provides real value** — intent classification over open-ended natural language is the justified case.
11. **CPU-friendly operation matters.** Small models, quantised ONNX, bounded context, no GPU requirement.
12. **Mira must support multiple Linux environments**, with the current GNOME + Wayland machine treated as a development environment, not the target.
13. **Every major component is independently testable**, without a microphone, a display, or a classifier.
14. **No giant monolithic classes and no duplicated responsibilities.** If two components can do the same thing, one of them is wrong.

**Explicit anti-goals for the next phases:** do not redesign TAMEV's architecture, do not replace working code for style, do not add a framework where a header plus a struct suffices, do not generate datasets or train models, and do not expand the intent set faster than the executor can support it.

---

## 5. Component Responsibilities

Each entry states what the component **owns**, what it **must NOT do**, and its interface at a sketch level. The "must NOT" lines are the architectural contract.

### 5.1 Audio / Microphone (`audio/`)

- **Owns:** microphone capture, sample format/rate normalisation (16 kHz mono float for Whisper), start/stop lifecycle, device selection, and (later) VAD / end-of-speech detection.
- **Must NOT:** transcribe, interpret, decide hotkeys, or know anything about intents.
- **Interface (target):** `bool start()`, `void stop()`, `std::vector<float> take()` (`take` transfers ownership of the buffer), `bool is_recording()`, plus a device-selection parameter.
- **Must-fix in Phase 14 (MUST):** synchronise shared state (atomics or a mutex/queue), make `stop()` actually quit the loop and join (no detached thread outliving the object), implement the destructor (disconnect/free/deinit), replace "first `Audio/Source` seen" with explicit device selection, replace the fixed 3-second window with VAD or hotkey-bounded capture.

### 5.2 Speech / Whisper (`speech/`)

- **Owns:** loading the ASR model, transcribing a float PCM buffer, exposing the raw transcript plus basic metadata (segment count, average confidence if available).
- **Must NOT:** decide intent, clean/normalise text for the tokenizer, or launch anything. Transcript normalisation belongs to `core/text_utils`.
- **Interface (target):** `Whisper(Config)`, `Transcription transcribe(const std::vector<float>& audio)` where `Transcription { std::string text; ... }`.
- **MUST:** model path comes from config/paths resolution, not a hardcoded relative string; language, thread count and model are configurable; failures return/propagate a `Result`, not a bare `throw` from deep inside the loop.

### 5.3 TAMEV (`tamev/`)

- **Owns:** project text + candidate options → probability distribution → selected `Intent` (+ confidence). It is an *option-selection* model, not a text generator, and it is the single authority on intent.
- **Must NOT:** extract the target/query/text/path (that is the tokenizer), execute anything, or hold per-intent extraction logic.
- **Interface (target):** `classifier.classify(context, utterance) → Decision { Intent intent; float confidence; std::vector<float> probs; }` with `probs` in `option_catalog` order.
- **MUST (Phase 6):** one canonical option ordering identical to training; one standard context format; a C++ WordPiece tokenizer loading `models/tamev-base/tokenizer.json`; and Python↔C++ parity tests before it is wired into the runtime. **Do not connect a potentially mismatched ONNX model blindly** — see Phase 6 and Phase 7.

### 5.4 Intent system (`core/intent.hpp`) — `MUST`, Phase 1

- **Owns:** the one and only definition of intent: `enum class Intent { OpenApplication, SearchWeb, ReadScreen, TypeText, FileOperation, SystemControl, Conversation, Unknown }`, its string form (`"open_application"`, lowercase snake_case, matching training), parsing/formatting, and the mapping to/from the model's option index.
- **Must NOT:** be duplicated. No component may define its own intent enum, string table, or option list. The uppercase string currently in `tokenizer.cpp` and the raw verb in `cmdmgr.cpp` both cease to exist as vocabularies.

### 5.5 Intent-aware Tokenizer (`tokenizer/`)

- **Owns:** given `(text, Intent)`, extracting the structured field(s) that intent needs — application reference, search query, text to type, file/path information, system-control verb/amount, screen-reader target.
- **Must NOT:** classify intent, resolve names to launch specs, execute anything, or become a general-purpose NLP/parser framework.
- **Interface (existing, keep):** `TokenResult process(const std::string& text, const std::string& intent)`; Phase 1/2 changes the parameter to `Intent` and returns the shared result type. The current private `extract_application` becomes one of several `extract_*` strategies selected by intent.
- **MUST (Phase 2):** `OPEN_APPLICATION` behaviour preserved, plus explicit "unsupported intent" handling and per-intent tests. Extension to other intents is incremental (Phase 8), never speculative.

### 5.6 Application Resolver (`resolver/`) — `MUST`, Phase 3

- **Owns:** loading `config/apps.json`, validating the schema/version, alias normalisation, and turning an application reference from the tokenizer into a concrete `LaunchSpec`. Matching strategy: exact name/alias → normalised (case/punctuation/spacing) → clearly-bounded fuzzy fallback, with an ambiguity result rather than a silent guess.
- **Must NOT:** launch, fork, exec, kill, or inspect running processes. It never touches the OS beyond reading its own configuration file.
- **Interface (target):** `ApplicationConfig::load(path) → Result<ApplicationConfig>`, `AppResolver::resolve(reference) → Result<LaunchSpec>`, `LaunchSpec { name; LaunchMethod method; command; path; working_directory; desktop_file; }`, `LaunchMethod { Command, Executable, Script, AppImage, Desktop }`.
- **MUST:** never invent a launch method for an entry that does not declare one; report "not configured" distinctly from "ambiguous" so the user can fix their config.

### 5.7 Application Manager (`appmgr/`) — `MUST`, Phase 4

- **Owns:** executing a `LaunchSpec` using the method it specifies, honouring `working_directory`, tracking the child process id, and reporting success/failure with the OS error.
- **Must NOT:** search for applications, read `apps.json`, guess aliases, or decide *whether* the user wanted an app launched. It receives a resolved spec and nothing else.
- **Interface (target):** `Result AppManager::launch(const LaunchSpec& spec)`, `Result AppManager::launch_dry_run(const LaunchSpec& spec)`, `std::vector<pid_t> running()`.
- **MUST:** all five methods (`command`, `executable`, `script`, `appimage`, `desktop`) implemented; `desktop` resolved via XDG data directories or `gio`/`xdg-open`, with the lookup documented as a *launch-mechanism detail*, not discovery of user intent; a dry-run/validation mode so tests never open real windows; correct argv/env/`argv[0]` handling and non-shell execution by default (no string concatenation into a shell).

### 5.8 Action Executor / Dispatcher (`executor/`) — `MUST`, Phase 5

- **Owns:** the mapping `Intent + TokenResult → concrete action`, invoking the correct collaborator (resolver/appmgr/input/screen), producing a single `Result`, and recording what was attempted for later verification.
- **Must NOT:** contain extraction logic, alias logic, or launching internals. It is a router plus per-intent action classes, never a God-object.
- **Interface (target):** `Result ActionDispatcher::execute(const ActionRequest& request)`; `ActionRequest { Intent intent; TokenResult tokens; std::string original_text; }`; one action class per intent under `executor/actions/`.
- **MUST:** an unhandled/`Unknown` intent produces a clear "not supported yet" `Result` rather than a crash or a silent no-op.

### 5.9 CommandManager (`cmdmgr/`) — **retire**, Phase 1/8

Current behaviour: lowercase + punctuation→space, split on whitespace, `action = tokens[0]`, `target = rest`, `valid = both non-empty`. It therefore *guesses intent from the first word* and duplicates the tokenizer's normalisation.

- **Decision:** its responsibility is split. Text normalisation moves to `core/text_utils`; intent guesswork is deleted (TAMEV owns it); target extraction is already the tokenizer's job. `CommandManager` is retained only as a temporary adapter during Phases 1–5 so `main.cpp` keeps working, then removed once the TAMEV → tokenizer → executor path handles `open_application` end to end (Phase 8). `Command`/`CommandManager` must not become part of the new core API.

### 5.10 Input Automation (`input/`) — `SHOULD`, Phase 9

- **Owns:** mouse move/click/drag, key press/release, text typing, hotkey chords — behind one interface with multiple backends.
- **Must NOT:** be the hotkey listener (that is a separate read-only class), nor read the screen, nor decide *what* to type.
- **Interface (target):** `InputBackend { backend_name(); bool available(); bool type_text(const std::string&); bool key(Key); bool hotkey(Chord); bool move_mouse(x,y,relative); bool click(Button); }` plus `InputBackendFactory::create_best()` using capability detection (env session type, portal presence, backend probes).
- **MUST:** X11/XTest and Portal/EIS backends are separate implementations; `ydotool`/`wtype` are optional external backends, never a permanent dependency; every action reports success/failure and never silently swallows a permission denial.

### 5.11 Screen Reader (`screen_reader/`) — `SHOULD`, Phase 10

- **Owns:** acquiring screen content (full screen, focused window, or region) and reporting *capability*: is capture available, is a permission prompt required, is window enumeration possible?
- **Must NOT:** interpret UI semantics (that is Vision), decide intents, or hardcode a desktop environment. GNOME-specific behaviour, if unavoidable, hides inside one backend.
- **Interface (target):** `SReader { std::string backend_name(); Capabilities capabilities(); Result<Image> capture(const CaptureRequest& req); }` plus `SReaderFactory::create_best()` and a static `probe_capabilities()`.
- **Planned backends:** `PortalBackend` (`org.freedesktop.portal.Screenshot` / screencast via `libportal`), `WaylandBackend` (compositor-specific protocols where they exist), `X11Backend` (XCB/XRandR).
- **MUST:** capability detection over assumption; on this machine the Portal backend is the realistic first implementation. Notable findings to plan around: **`grim` is wlroots-only and will not work on GNOME Wayland**, and **GNOME does not expose a window list to portals**, so "which window is focused / enumerate windows" cannot be assumed (see the Phase 12 risk).

### 5.12 Vision / Screen Understanding (`vision/`) — `FUTURE`, Phase 11

- **Owns:** turning an image into structured understanding: OCR text with regions, coarse UI element detection, and (later) app/window state classification.
- **Must NOT:** capture the screen (that is `SReader`) or contain any intent logic. OpenCV is treated as an image-processing *toolkit*, not as a UI-understanding system.
- **Interface (target):** `ScreenAnalysis analyze(const Image&)`; a separate `OcrBackend` interface so the OCR engine can be swapped. **Decision deferred:** `tesseract` is not installed — no OCR engine should be assumed yet.

### 5.13 Verification (`verification/`) — `FUTURE`, Phase 12

- **Owns:** answering "did the action actually work?" using whatever evidence is available (process existence, screen delta, OCR text, window title where obtainable), plus a bounded retry/timeout policy.
- **Must NOT:** perform the action, retry indefinitely, or block the assistant on an unavailable capability — it must return "unverified" distinctly from "failed".
- **Interface (target):** `VerificationResult verify(const ActionRecord& record, const VerificationPolicy& policy)`; strategies are pluggable per action type.

### 5.14 TTS (`tts/`) — `SHOULD`, Phase 13

- **Owns:** converting a response string to speech, with backend selection and start/stop control.
- **Must NOT:** create the response text (the executor/responder owns phrasing), or block the main loop without a control path.
- **Interface (target):** `TtsEngine { bool say(const std::string&); void stop(); bool is_speaking(); }` with a subprocess backend first (`piper` or `espeak-ng`). **Decision deferred:** no TTS engine is installed yet; choose one when Phase 13 starts and treat it as an external dependency like Whisper.

### 5.15 Cross-cutting: configuration, paths, logging (`core/`) — `SHOULD`, Phases 0/1

- **Owns:** locating and loading Mira's own configuration (model paths, hotkey chord, audio device, TTS voice, backend preferences), path resolution independent of CWD, and minimal levelled logging.
- **Must NOT:** parse `apps.json` (resolver's job), or scatter per-component defaults across the codebase.
- **Rationale from analysis:** `main.cpp` hardcodes a relative model path and `hotkey.cpp` hardcodes `/dev/input/event3`; both are symptoms of missing configuration and path resolution.

---

## 6. Data Flow

### 6.1 Intended runtime flow

```
[1] HotkeyListener        user presses the activation chord
        │
[2] Microphone            capture utterance audio (16 kHz mono float)
        │
[3] Whisper               transcript: "open firefox"
        │
[4] TAMEV                 context + options → intent = open_application (conf 0.94)
        │                 ← INTENT IS DECIDED HERE, BEFORE ANY EXTRACTION
        │
[5] Tokenizer             (text, open_application) → TokenResult{ TARGET, "firefox" }
        │
[6] Executor              ActionRequest{ intent, tokens } → dispatch
        │
[7] Resolver              "firefox" → LaunchSpec{ method = command, command = "firefox" }
        │
[8] AppManager            launch(LaunchSpec) → pid
        │
[9] Verification          did firefox appear?  (Phase 12)
        │
[10] Responder / TTS      "Opening Firefox" → spoken
```

Ordering is a hard requirement: steps 4 and 5 may not be swapped, and step 5 may never run without an intent from step 4.

### 6.2 Worked examples

| User utterance | TAMEV intent | Tokenizer output | Executor path |
|---|---|---|---|
| "Open Firefox" | `open_application` | `TARGET = "Firefox"` | resolver → appmgr |
| "Search the internet for how transformers work" | `search_web` | `QUERY = "how transformers work"` | search/browser action |
| "Type hello world into the editor" | `type_text` | `TEXT = "hello world into the editor"` (target refinement is Phase 8) | input backend |
| "Move my essay to Documents" | `file_operation` | `PATH` + destination fields | file action (**FUTURE** — needs the safety policy in §12) |
| "Turn the volume down" | `system_control` | command + amount | system action |
| "What does this window say?" | `read_screen` | relevant target/context | screen reader → vision → TTS |

### 6.3 Failure paths that must be designed, not discovered

```
Whisper produces empty/garbage text    → Rejected("I didn't catch that")     ; no TAMEV call
TAMEV confidence below threshold       → Intent::Unknown                    ; ask, do not act
Tokenizer returns invalid/empty field  → Rejected("what should I open?")     ; no resolver call
Resolver: not configured               → Rejected("Firefox isn't in apps.json")
Resolver: ambiguous match              → Rejected(list of candidates)
AppManager: exec fails (ENOENT/EACCES) → Failed(os error text) + log entry
Backend unavailable / permission denied→ Unavailable(...) → spoken explanation, no retry loop
Verification inconclusive              → Unverified → report honestly, never claim success
```

Every arrow above crosses a stable interface and returns a `Result`; no component may swallow a failure silently or substitute a fallback action the user did not ask for.

---

## 7. Development Phases

Every phase must: build successfully, keep existing behaviour working, add tests, be verified manually, and stop for review before the next phase. Order matters; do not start a phase before its dependencies are green.

### Phase 0 — Project Safety & Build Foundation

**Priority:** `MUST`

- **Objective:** make the project reproducible, recoverable, and warning-clean before adding architecture.
- **Components involved:** repository root, `CMakeLists.txt`, `build/`, `speech/whisper_lib/`, and the legacy prototype location outside the project (`/home/arc/mira-tools/live.cpp`).
- **Tasks:**
  1. Initialise version control (`git init` in `miraV2`) and add a `.gitignore` excluding `build/`, `.cache/`, model blobs and binaries. *(Note: no VCS exists today — this is the single highest-risk gap.)*
  2. Add a baseline commit of the current state before any refactor.
  3. Set `CMAKE_BUILD_TYPE` (default `RelWithDebInfo`), enable `-Wall -Wextra`, and record the warnings that already exist in `audio/microphone.cpp` as known baseline.
  4. Add a test target structure (`enable_testing()`, `tests/`, `add_test`) with one trivial passing test to prove the wiring.
  5. Decide and fix the Whisper library policy so the build is self-contained: either vendor the ggml libraries alongside `libwhisper.so.1` and neutralise the `RUNPATH` pointing at `/home/arc/mira-tools/whisper.cpp/build/bin`, or build whisper.cpp as a proper external project/submodule. Document the choice.
  6. Settle the legacy prototype intentionally. It is no longer in the project root; an identical copy exists at `/home/arc/mira-tools/live.cpp`. If it is worth keeping, bring it back under `experiments/` (excluded from the build) with a header stating it is not wired in; if it stays outside the project, record it here as historical. Either way, do not repeat its duplication of PipeWire capture and Whisper integration in the new architecture. Do not commit build outputs or the hand-built `tokenizer/test_tokenizer` binary.
  7. Document the build commands in a short `BUILDING.md` section inside this roadmap or a `README` addition (do not restructure beyond that).
- **Dependencies:** none.
- **Tests required:** `cmake -S . -B build && cmake --build build` succeeds from a clean build directory; `ctest` runs and passes; `ldd build/miraV2` shows no dependence on paths outside the project.
- **Completion criteria:** repository under version control; clean rebuild works; tests runnable; the external-path fragility is resolved or explicitly documented as a known, tracked issue.
- **Risks:** breaking the currently working binary while touching linking; deciding to vendor libraries increases repository size (models are already large and should stay out of git history — plan for `.gitignore` + an explicit `models/README` describing required downloads).

### Phase 1 — Core Architecture

**Priority:** `MUST`

- **Objective:** establish the canonical vocabulary and interfaces everything else will use.
- **Components involved:** new `core/`, existing `tokenizer/`, existing `cmdmgr/`, `main.cpp`.
- **Tasks:**
  1. Create `core/intent.hpp/.cpp`: `enum class Intent` with the eight values, `to_string`/`from_string` in lowercase snake_case (matching the training options), and `Intent ↔ option_index` helpers.
  2. Create `core/result.hpp`: a `Result` type with status (`Ok`, `Rejected`, `Unavailable`, `Failed`, `Unverified`) plus message/detail, so components return failures instead of throwing across boundaries.
  3. Create `core/action.hpp`: `ActionRequest { Intent; TokenResult; std::string original_text; }` and a shared `TokenResult`/`TokenType` (moved from `tokenizer.hpp`, which then includes `core`).
  4. Create `core/text_utils.hpp/.cpp` and move normalisation/trim there, reimplementing nothing: the logic already present in `cmdmgr.cpp` and `tokenizer.cpp` becomes one shared implementation.
  5. Create `core/paths.hpp/.cpp` for resolving config/model paths independent of the current working directory.
  6. Change `Tokenizer::process` to take `Intent` instead of a free-form string; keep behaviour for `OpenApplication` identical.
  7. Convert `CommandManager` into a temporary, clearly-marked adapter that no longer defines its own action vocabulary, and mark it for removal in Phase 8.
  8. Restructure CMake into a `mira_core` library plus the existing executable (no functional change), so tests can link the core without the binary.
- **Dependencies:** Phase 0.
- **Tests required:** unit tests for intent string round-trips (all eight values, including unknown input), text-utils normalisation against the current `cmdmgr` behaviour, and tokenizer output unchanged for the existing `OPEN_APPLICATION` cases.
- **Completion criteria:** exactly one intent vocabulary in the codebase; the app still runs the old flow end to end; core builds as a library and is unit-testable.
- **Risks:** scope creep into a "framework"; introducing a `Result` type that is too elaborate. Keep it to what the executor actually needs.

### Phase 2 — Tokenizer Foundation

**Priority:** `MUST`

- **Objective:** make the tokenizer a first-class, tested, intent-aware component that runs *after* an intent is known.
- **Components involved:** `tokenizer/`, `core/`, CMake, `tests/`.
- **Tasks:**
  1. Bring `tokenizer/` into the CMake build as part of `mira_core`; make `test_tokenizer.cpp` a `ctest` case instead of a hand-compiled binary.
  2. Keep the architecture order explicit in the API: `process(const std::string& text, Intent intent)` — the intent parameter is required, never defaulted, so a caller cannot extract before classifying.
  3. Preserve the current `OpenApplication` behaviour exactly (case-insensitive removal of a leading `open`/`launch`/`start`/`run`, trim, non-empty validation), then re-express it on top of `core/text_utils`.
  4. Add explicit handling for unsupported intents: return an invalid/typed result with a clear reason rather than an empty `UNKNOWN`.
  5. Define the extraction contract for the remaining intents (query, text, path, system verb) as documented structure without implementing them yet — a one-screen design section, not code.
  6. Move normalisation out of `cmdmgr` into `core/text_utils` usage (finishes the Phase 1 split).
- **Dependencies:** Phase 1.
- **Tests required:** table-driven tests for `OpenApplication` covering all four verbs, mixed case, extra whitespace, punctuation, multi-word names ("open Unity Editor"), missing target ("open"), empty input, and target text that merely contains a verb word. Plus a test proving an unsupported intent yields a defined failure rather than garbage.
- **Completion criteria:** tokenizer in the build, exercised by `ctest`, `OpenApplication` behaviour unchanged and pinned by tests, and the intent parameter is structurally enforced.
- **Risks:** turning the tokenizer into an NLP system (do not — it is a per-intent extractor); over-normalising text so multi-word application names are damaged; silently changing existing behaviour that the rest of the pipeline will rely on.

### Phase 3 — Application Configuration & Resolver

**Priority:** `MUST`

- **Objective:** resolve a user-spoken application reference into a concrete launch specification, using user configuration.
- **Components involved:** new `resolver/`, new `config/apps.json`, `core/` (`Result`, `text_utils`), CMake.
- **Tasks:**
  1. Define the JSON schema exactly as intended: `version`, `applications[]` with `name`, `aliases[]`, `launch { method, command|path|desktop_file, working_directory }`, with the five methods (`command`, `executable`, `script`, `appimage`, `desktop`).
  2. Use the already-installed `nlohmann/json` (3.12.0, header-only) — no new dependency and no custom parser writing.
  3. Implement `ApplicationConfig::load()` with schema validation: unknown version, missing required fields per method, duplicate names/aliases, malformed JSON → distinct, actionable errors.
  4. Implement `LaunchSpec` as the sole handoff type to `AppManager`, including the launch method enum.
  5. Implement `AppResolver::resolve()` with a documented matching order: exact name/alias → normalised (case, punctuation, spacing) → bounded fuzzy fallback; return "not configured" vs "ambiguous" distinctly.
  6. Create a small default `config/apps.json` containing a few well-understood entries for the development machine (for example the browsers/editors actually present) plus the documented example entries from the project description. **No application may be launched in this phase.**
- **Dependencies:** Phase 1.
- **Tests required:** config loader tests (valid file, each of the five methods, all invalid variants); resolver tests (exact, alias, normalised, ambiguous, not found, empty reference); a test asserting resolution performs no side effects.
- **Completion criteria:** for a fixed `apps.json`, spoken references resolve to the expected `LaunchSpec` with a high degree of determinism; all configuration and resolution paths covered by tests.
- **Risks:** fuzzy matching producing confident wrong answers (keep it bounded, prefer "ambiguous"/"not configured"); schema drift between the C++ loader and the future Tkinter editor (treat the schema as a versioned contract); embedding machine-specific paths in committed defaults.

### Phase 4 — AppManager

**Priority:** `MUST`

- **Objective:** actually launch applications from a resolved `LaunchSpec`, safely and testably.
- **Components involved:** `appmgr/` (currently 0-byte placeholders), `resolver/` types, `core/Result`, CMake.
- **Tasks:**
  1. Implement `AppManager::launch(const LaunchSpec&)` for all five methods: `command` (program + args via `PATH`), `executable` (absolute path), `script` (shell-interpreted script at an absolute path), `appimage` (AppImage path, executable bit handling), `desktop` (XDG `.desktop` lookup and launch).
  2. Use `posix_spawn`/`fork`+`execve` with an explicit `argv`/`envp` — never concatenate user text into a shell command string.
  3. Honour `working_directory` when specified; define behaviour when it does not exist (reject with a clear error).
  4. Detach intentionally: Mira should not become the parent that reaps the child indefinitely; record pid(s) for later verification, and document the chosen reaping/`SIGCHLD` policy.
  5. Add `launch_dry_run()`/validation used by tests: resolves paths, checks existence/permissions, prints the intended `execve` arguments, launches nothing.
  6. Return `Result` with the OS error on failure (`ENOENT`, `EACCES`, `ENOTDIR`, bad shebang, etc.).
- **Dependencies:** Phase 3 (needs `LaunchSpec`).
- **Tests required:** dry-run tests for every method and every failure mode against temporary fixtures created by the test (a script in `/tmp`, a dummy executable, a fake `.desktop` file). A single opt-in manual test for a harmless real launch (for example a GUI-less command that writes a file) that is not enabled by default in CI.
- **Completion criteria:** all five methods work by dry-run/validation; at least one real harmless launch verified manually; no shell string construction anywhere.
- **Risks:** accidental launching of GUI applications during automated tests (mitigate: dry-run is the default for tests); zombie processes; environment differences (`PATH`, `DISPLAY`, `WAYLAND_DISPLAY`) when launching from a hotkey-started process; AppImage needing FUSE or the `--appimage-extract-and-run` fallback (document, do not silently change behaviour).

### Phase 5 — Action Executor

**Priority:** `MUST`

- **Objective:** connect `Intent + TokenResult` to a concrete action and produce a single `Result`, with `OpenApplication` working end to end.
- **Components involved:** new `executor/`, `resolver/`, `appmgr/`, `tokenizer/`, `core/`.
- **Tasks:**
  1. Implement `ActionDispatcher::execute(const ActionRequest&)` with an explicit dispatch table and one action class per intent under `executor/actions/`.
  2. Implement `OpenApplicationAction`: tokenizer target → `AppResolver::resolve` → `LaunchSpec` → `AppManager::launch` → `Result`. It contains no alias or exec logic of its own.
  3. Implement the `Unknown`/unsupported-intent path returning "not supported yet".
  4. Record an `ActionRecord` (intent, extracted value, resolution, outcome, timestamp) for later verification and logging.
  5. Establish the temporary manual-entry path needed to test before TAMEV exists: intent supplied on the command line or via a prompt, clearly marked as scaffolding to be removed in Phase 8.
- **Dependencies:** Phases 1, 2, 3, 4.
- **Tests required:** dispatcher unit tests with a stubbed resolver/manager (verify routing and `Result` propagation, no real launching); an integration test of the action chain against a fixture config in dry-run mode; one manual end-to-end run accepting typed input and launching a harmless configured application.
- **Completion criteria:** the vertical slice `typed text → tokenizer → resolver → AppManager (real, harmless) → Result` works reliably, and TAMEV is still mocked or manual. This is the first point at which Mira *does* something.
- **Risks:** pulling real extraction logic into the action (keep it in the tokenizer); a dispatcher that grows into a God-object as intents are added (hence one class per intent); scaffolding paths leaking into the final runtime (delete them in Phase 8).

### Phase 6 — TAMEV C++ Integration

**Priority:** `MUST`

- **Objective:** run TAMEV intent classification inside the C++ runtime with results matching Python.
- **Components involved:** new `tamev/`, `core/` (`Intent`, option mapping), ONNX Runtime, `models/`, CMake.
- **Tasks:**
  1. Add ONNX Runtime to the build via `find_package(onnxruntime)` (1.29 is installed with a CMake config) and keep it optional so the app still builds and tests without a model.
  2. Implement a C++ WordPiece tokenizer: parse `models/tamev-base/tokenizer.json` (WordPiece, 30522 entries, BERT normaliser, lowercase) with `nlohmann/json`. Note there is **no `vocab.txt`** to rely on. Keep the name clearly distinct from the project's intent-aware `Tokenizer` (for example `BertWordPieceTokenizer`) to avoid the existing naming collision.
  3. Implement `TamevClassifier` over the confirmed ONNX contract: `ctx_input_ids`/`ctx_attention_mask` `(B, ctx_len)`, `opt_input_ids`/`opt_attention_mask` `(B, K, opt_len)` as `int64`, outputs `logits`/`probs` `(B, K)` `float32`. Confirm the actual dynamic shapes at load time rather than assuming fixed `ctx_len = 128` / `opt_len = 64`.
  4. Define `tamev/option_catalog.hpp` as the single candidate list, in the exact training order (order is a hard contract because labels are `OPTIONS.index(label)`), and map index → `core::Intent`.
  5. Define exactly one standard context format and option padding, resolving the current `tester.py` vs `train.py` divergence. Document it in one place and use it from both languages.
  6. Add a confidence threshold: below it, return `Intent::Unknown` rather than a low-confidence wrong action.
  7. Build a parity harness: run a fixed list of utterances through the Python reference (the `models/tester.py` inference style) and through C++, comparing argmax *and* probability vectors.
- **Dependencies:** Phase 1 (intent enum). **Model-correctness note:** the only ONNX file present is the *downloaded base* artifact, not an export of `mira-tamev`. Phase 6 should therefore be read as "build the inference path and prove parity against a fixed artifact"; wiring a *fine-tuned* model is gated on Phase 7. Phases 6 and 7 may be executed in either order, but **Phase 8 requires both.**
- **Tests required:** tokenizer unit tests (word splitting, unknown tokens, truncation, `[CLS]`/`[SEP]` insertion, lowercase handling) validated against Python output; classifier tests on a frozen utterance set compared to Python within a tight tolerance; a test asserting `probs.size() == K` and that `option_catalog` order maps to the intended `Intent` values; a test that a missing/corrupt model returns `Unavailable` without crashing.
- **Completion criteria:** C++ and Python agree on intent and probability ordering for the frozen set; no hardcoded option list or context string appears anywhere else in the codebase.
- **Risks:** the largest technical risk in the project. Mismatched option ordering, wrong context format, wrong padding, `int64` vs `int32` inputs, differing truncation, and the Python 1.30 / system 1.29 runtime skew can all produce silently wrong probabilities. Mitigation: parity tests are mandatory before any wiring, and probabilities — not only the argmax — are compared.

### Phase 7 — Training / Data Pipeline Fix

**Priority:** `MUST` for model quality (no C++ runtime code is written in this phase)

- **Objective:** make the training pipeline consume the datasets that actually exist, and produce a reproducible, correctly-exported model.
- **Components involved:** `models/training/`, `models/tester.py`, `models/runners/tester.py`, `models/training/datasets/`, ONNX export.
- **Tasks:**
  1. Connect the existing generated datasets (`datasets/open_application.jsonl` = 1165 rows, `datasets/search_web.jsonl` = 1016 rows, 2181 examples total) to `train.py`. Today it trains on the 56/7/7 split of the 70-example `dataset.py`, which is the likely cause of the ~57 % accuracy.
  2. Apply balanced train/validation/test splits with deduplication and leakage checks (no near-duplicate text across splits), stratified per intent.
  3. Report per-intent precision/recall and a confusion matrix rather than a single accuracy figure, which currently hides which intents fail.
  4. Perform error analysis before deciding whether more data is needed. **Do not generate new datasets in this phase beyond what is needed to close evident label gaps.** Quality dimensions that matter: diversity, natural-language variation, realistic commands, different word orders, short and long commands, casual speech, imperfect English, ambiguity, hard negatives, balanced intents, no duplicates, correct labels.
  5. Add the missing reproducible export step from the fine-tuned model to ONNX: scripting/tracing, dynamic axes for `batch`, `ctx_len`, `opt_len`, opset choice, int8 quantisation, plus a verification run of the exported file against PyTorch outputs. **Nothing in the repository produces ONNX today.**
  6. Record the exact export recipe and the parity-harness invocation so any future retrain yields a C++-compatible artifact.
- **Dependencies:** none strictly, but it must precede the point where a fine-tuned model is placed into the runtime (second half of Phase 8). Uses `/home/arc/ml_env` (`torch 2.13.0+cu130`, `transformers 5.14.1`, `onnxruntime 1.30.0`).
- **Tests required:** dataset integrity checks (schema, valid labels, duplicate detection, split disjointness, class-balance report); export verification (ONNX vs PyTorch maximum absolute difference on a fixed sample); the Phase 6 parity harness re-run against the newly exported artifact.
- **Completion criteria:** the intended datasets are consumed; per-intent metrics are reported; the exported model measurably beats the previous one; the export is reproducible from a documented command.
- **Risks:** the 2181 examples are skewed (1165 `open_application` vs 1016 `search_web`, with the other five intents absent from `datasets/`), so training on them alone unbalances the 7-class task — fix with balancing and coverage rather than by inflating volume; int8 quantisation tolerance; overfitting to templated phrasing in generated data (mitigate with realistic variation and hard negatives).

### Phase 8 — Full TAMEV → Tokenizer → Executor Pipeline

**Priority:** `MUST`

- **Objective:** remove the scaffolding and run the real pipeline, then widen intent coverage one intent at a time.
- **Components involved:** `main.cpp`, `tamev/`, `tokenizer/`, `executor/`, `resolver/`, `appmgr/`, `speech/`, `audio/`, `input/hotkey`.
- **Tasks:**
  1. Wire `Whisper → TAMEV → Intent → Tokenizer → Executor` and delete the Phase 5 manual-entry scaffolding, the `CommandManager` adapter, and any temporary intent strings.
  2. Add an intent confidence gate and a "did I hear that right?" clarification path for `Unknown`.
  3. Reliable `open_application` first: verified on real spoken input for command/executable/script/desktop/AppImage entries.
  4. Then extend, one intent per increment, each completing tokenizer extraction + executor action + tests before the next: `search_web` (browser/search action), `type_text` (needs Phase 9), `system_control`, `read_screen` (needs Phase 10), `file_operation` (needs §12 safety policy), `conversation` (no system action; routing only).
  5. Add end-to-end logging of the full decision chain: transcript → intent + probabilities → extracted token → resolution → outcome.
- **Dependencies:** Phases 2, 5, 6 (and Phase 7 for a fine-tuned model), plus Phase 9 for `type_text` and Phase 10 for `read_screen`.
- **Tests required:** end-to-end tests driving the pipeline with **text input** (bypassing the microphone) so they are deterministic and CI-friendly, covering one happy path and one failure path per supported intent; a regression test that a low-confidence classification never triggers an action.
- **Completion criteria:** the intended pipeline runs end to end for `open_application` with real speech, and any other intent marked "done" has its own tests; no scaffolding remains.
- **Risks:** extending the intent set faster than the executor supports (the roadmap explicitly forbids this); microphone flakiness masking pipeline bugs (use text-mode tests); unclear responsibility creep between tokenizer and action classes as new intents are added.

### Phase 9 — Input Automation

**Priority:** `SHOULD` (needed for `type_text`; not needed for `open_application`)

- **Objective:** provide a backend-abstracted way to type text, press keys, use hotkeys, move the mouse and click.
- **Components involved:** `input/`, `executor/` (`TypeTextAction`), `core/Result`.
- **Tasks:**
  1. Define the `InputBackend` interface (see §5.10) plus a `probe`/`available()` mechanism and a factory that selects the best backend.
  2. Implement capability detection by probing, not by assuming the desktop: session type, portal presence on the bus, backend self-test. The current machine is GNOME + Wayland, where the realistic routes are the RemoteDesktop portal + EIS (`libei`/`liboeffis` 1.6.0 are installed) or a compositor-specific path. X11/XTest is the second backend (`libX11`/`libXtst` bindings via `x11`/`xcb`).
  3. Keep `ydotool`/`wtype` as **optional external** backends only — neither is installed, and neither is permitted to become a hard dependency.
  4. Handle the unavoidable interactive permission prompt as a first-class outcome: report `Unavailable`/"permission required" clearly instead of hanging or silently failing.
  5. Implement `TypeTextAction` on top of the backend, then wire `type_text` into the executor.
  6. Define a test mode that records intended events (a "dry-run backend") so automation can be unit-tested without touching the live session.
- **Dependencies:** Phase 1; Phase 5 for the executor pattern.
- **Tests required:** unit tests of key/character mapping (ASCII, shifted characters, non-ASCII/Unicode limitations stated explicitly), dry-run backend tests for `TypeTextAction`, and a manual verification step per backend on the development machine.
- **Completion criteria:** at least one working backend on the current session, a second backend defined behind the same interface, a dry-run backend for tests, and honest reporting when injection is unavailable.
- **Risks:** Wayland input injection is the second-largest portability risk after screen capture — EIS/portal behaviour is version- and compositor-dependent, and permission prompts may require user interaction; Unicode/BMP handling differs per backend; injecting keystrokes into the wrong window is a real safety issue (see §12).

### Phase 10 — Screen Capture

**Priority:** `SHOULD`

- **Objective:** capture the screen (or a region/window where possible) through a backend abstraction with capability detection.
- **Components involved:** `screen_reader/` (currently 0-byte placeholders), `core/Result`, CMake.
- **Tasks:**
  1. Define `SReader` and `Capabilities` (see §5.11): capture availability, permission requirement, window enumeration support, region support.
  2. Implement `PortalBackend` first using the installed `libportal` 0.11.0 (`org.freedesktop.portal.Screenshot`, screencast where needed). `xdg-desktop-portal-gnome` is present on the development machine.
  3. Implement `X11Backend` (XCB/XRandR) for X11 sessions, kept entirely separate from the Wayland path.
  4. Define the `WaylandBackend` slot honestly: implement only where a real protocol is available, and return `Unavailable` where none is — do **not** pretend `grim` works on GNOME (it is wlroots-only).
  5. Isolate any GNOME-specific behaviour inside a single backend file with a comment explaining why it exists.
  6. Store captures in memory as a simple image type with metadata (dimensions, scale factor, timestamp); do not add an image framework — `fromImage`-style handoff to `vision/` is enough for now.
- **Dependencies:** Phase 1.
- **Tests required:** capability-probe tests that run headless and assert "unavailable" without a session; a capture test that is skipped (not failed) when no display/portal is available; a manual capture verification on the development machine.
- **Completion criteria:** at least one backend captures successfully on the current machine; capability probing works and returns `Unavailable` gracefully on machines where capture is impossible; no compositor-specific code outside backends.
- **Risks:** portal permission prompts (interactive, possibly not automatable); HiDPI scaling and multi-monitor geometry; expensive captures (large buffers, latency) — define a bounded resolution/region policy; screenshots are sensitive data (§12).

### Phase 11 — Screen Understanding / Vision

**Priority:** `FUTURE` (must not block Phases 1–10)

- **Objective:** derive structure from a captured image: OCR text with regions, coarse UI element detection, and (later) application/window state.
- **Components involved:** new `vision/`, `screen_reader/`, OpenCV.
- **Tasks:**
  1. Define an `OcrBackend` interface and *choose* an engine when this phase starts — `tesseract` is **not installed**, so this is an explicit, deferred technology decision, not an assumption.
  2. Implement basic preprocessing with OpenCV where genuinely useful (scaling, thresholding, region cropping). Treat OpenCV as a toolkit, not as UI understanding.
  3. Define `ScreenAnalysis` as the output contract: text blocks with bounding boxes, candidate UI elements, and confidence, kept stable so later models can replace the implementation.
  4. Keep any ML vision model behind its own interface, loaded separately from the screen-capture path, so capture works without it.
  5. Wire `read_screen` only far enough to answer "what does this say?" textually; defer interaction/clicking-by-vision.
- **Dependencies:** Phase 10 (images), Phase 1 (contracts).
- **Tests required:** golden-image tests on small fixture images (synthetic text rendered by the test) with tolerance for OCR variation; a test that an empty/failed analysis returns a defined result rather than throwing.
- **Completion criteria:** OCR text extraction works on a captured screenshot of a known fixture; the contract allows swapping OCR engines; capture continues to work with vision absent.
- **Risks:** OCR accuracy on UI text varies wildly with scaling, theme and anti-aliasing; false confidence leading to wrong actions; heavyweight vision models violating the CPU-friendly principle (keep them optional and off the critical path).

### Phase 12 — Verification

**Priority:** `FUTURE` (developed after the execution foundation exists)

- **Objective:** determine whether a requested action actually succeeded, and report honestly when it cannot be determined.
- **Components involved:** new `verification/`, `screen_reader/`, `appmgr/`, `executor/`.
- **Tasks:**
  1. Define pluggable verification strategies per action type: process/child exists (`open_application`), screen delta / expected text present (`type_text`, `search_web`), explicit failure detection (error dialog/crash).
  2. Define `VerificationResult` with three distinct outcomes: `Verified`, `Failed`, `Unverified` — never collapse `Unverified` into `Failed`, and never report `Verified` from an absence of evidence.
  3. Implement bounded retries with timeouts and a documented policy (how long to wait, how many attempts, what to do on timeout).
  4. Start with the cheapest reliable evidence: for `open_application`, a recorded pid plus a bounded liveness/`/proc`-based check; add screen-based evidence only once Phase 10 exists.
  5. Connect the outcome into the response/TTS layer so Mira says "I opened it" only when verified.
- **Dependencies:** Phases 5 and 10 (screen evidence), Phase 13 optional (to *tell* the user the outcome).
- **Tests required:** unit tests with injected fake evidence for all three outcomes; timeout/retry tests with a controllable clock or injected delay; a "no evidence available" test asserting `Unverified`.
- **Completion criteria:** `open_application` verification works without screen access; screen-based verification is available where capture is available; no path reports success without evidence.
- **Risks:** **the highest architectural risk in this roadmap.** On GNOME Wayland there is no portable window list or window-title access, so "did the app appear?" cannot rely on window enumeration; a portal screencast plus pixel/OCR heuristics is the fallback but is fragile, and the GNOME extension/D-Bus route is not portable. Plan verification to be *best-effort and honest*, keep it pluggable, and never let it block the assistant. Also avoid flakiness (animations, timing, resolution) making verification untrustworthy.

### Phase 13 — TTS

**Priority:** `SHOULD`

- **Objective:** let Mira speak responses through a replaceable backend.
- **Components involved:** new `tts/`, `executor/`/responder, `core/Result`.
- **Tasks:**
  1. Define `TtsEngine` (see §5.14) with `say`, `stop`, `is_speaking`.
  2. Implement one lightweight backend as a managed subprocess. **Choose the engine at the start of this phase** — no TTS engine is installed today, so this is a deferred decision (candidates: `piper` for quality, `espeak-ng` for size/simplicity).
  3. Manage process lifecycle properly: no zombies, cancel support, and a defined behaviour if the engine is missing (report `Unavailable`, never crash or hang).
  4. Add a response-phrasing layer that turns `Result` into short spoken sentences ("Opening Firefox", "I don't have that application configured"), keeping phrasing out of the executor.
  5. Make TTS optional at build *and* runtime so the pipeline works silently in headless tests.
- **Dependencies:** Phases 5 and 8 (something worth saying).
- **Tests required:** engine-absent test returning `Unavailable`; a fake/mock backend test asserting the correct phrase is produced for each `Result` status; a test asserting TTS never blocks pipeline completion indefinitely.
- **Completion criteria:** Mira speaks action results with a replaceable backend, works with TTS disabled, and cannot be blocked by a stuck TTS process.
- **Risks:** blocking the main loop on speech; audio-device contention with the microphone (a common and annoying failure mode); an external dependency that may be missing on other machines (must degrade gracefully).

### Phase 14 — Reliability / Portability

**Priority:** `MUST` for the hardening items, `SHOULD` for the rest

- **Objective:** remove the machine-specific and lifecycle bugs identified in the analysis, and make Mira workable on more than one Linux environment.
- **Components involved:** `audio/microphone.*`, `input/hotkey.*`, `speech/whisper.*`, `core/paths`, `core/log`, config, CMake.
- **Tasks:**
  1. **Fix the hotkey listener:** remove the hardcoded `/dev/input/event3`; discover the keyboard device (by-name/by-id lookup, capability check for `EV_KEY`/keyboard keys, or an explicit configured device); never `read()` a negative fd (no busy-spin); make the activation chord configurable; report a clear error if no keyboard device is accessible.
  2. **Fix the microphone lifecycle:** replace the detached thread with an owned, joinable thread; make `stop()` quit the main loop and join; implement the destructor (stream disconnect, main-loop/context destroy, `pw_deinit`); make shared state safe (atomics or a queue) so `get_audio()`/`take()` is race-free; replace "last `Audio/Source` seen" with explicit device selection; replace the fixed 3-second capture with hotkey-bounded capture and/or VAD.
  3. **Make paths CWD-independent:** model, config and log paths resolved through `core/paths` (fixes the relative `models/ggml-base.en.bin`).
  4. **Add configuration and logging:** a Mira config file (audio device, hotkey chord, model paths, backend preferences, confidence threshold) plus levelled logging that never prints to stdout in the middle of the interaction loop.
  5. **Portability sweep:** replace remaining desktop assumptions with capability detection; ensure backend-dependent features degrade to `Unavailable` instead of failing obscurely; verify the CMake options (`MIRA_ENABLE_X11`, `MIRA_ENABLE_PORTAL`, `MIRA_ENABLE_ONNX`, TTS) allow a headless build.
  6. **Resource cleanup review:** no leaked fds, threads, subprocesses or ONNX sessions on the error paths; every failure path releases what it acquired.
  7. **Add regression tests** for each fixed bug (for example a hotkey test asserting no busy-spin without a device, a microphone test asserting a clean start/stop/restart cycle).
- **Dependencies:** Phase 0 (test harness); touches everything from Phases 1–13.
- **Tests required:** regression tests per fix; a "no display / no model / no TTS" run that still starts and reports `Unavailable` correctly; repeated start/stop cycles without fd or thread growth (a leak check on the audio path).
- **Completion criteria:** no hardcoded device paths; clean start/stop/restart; no data races in the audio path (verified with a sanitiser run where feasible); Mira starts and degrades gracefully on a machine missing optional backends.
- **Risks:** the microphone fixes touch the only interactively-manual component (hard to unit test), so changes must be verified with real recording; VAD introduces new failure modes (truncated speech, never-ending capture) — keep a hard maximum duration as a watchdog.

### Phase 15 — Final Mira Integration

**Priority:** `SHOULD`/`FUTURE` (the integration milestone)

- **Objective:** a complete, usable voice loop across multiple intents with honest, robust failure handling.
- **Components involved:** all.
- **Tasks:**
  1. Full voice loop: hotkey → capture → transcript → intent → extraction → action → verification → spoken response, with the loop idempotent and interruptible.
  2. Enable and harden the intent set that has real executors: `open_application`, `search_web`, `type_text`, `system_control`, `read_screen`, and `conversation` routing.
  3. Wire verification outcomes and TTS together so Mira reports truthfully (including `Unverified` and permission-denied cases).
  4. Failure handling pass: every failure mode in §6.3 exercised deliberately, with spoken explanations that a user can act on.
  5. Configuration/UX touchpoints: `config/apps.json` documentation, first-run sanity checks (models present, backends available), and a clear startup report of what is and is not available.
  6. Optional GUI configuration utility (Python/Tkinter) for editing `apps.json`, treated as a separate tool that writes the same versioned schema — **not** part of the C++ runtime.
- **Dependencies:** everything, especially Phases 8, 9, 10, 12, 13, 14.
- **Tests required:** a full end-to-end test matrix in text mode (intent × action, happy path + failure path), a manual voice-session checklist per release, and a regression suite that runs headless.
- **Completion criteria:** the §13 Definition of Done is satisfied.
- **Risks:** integration reveals contract mismatches late (mitigate by keeping interfaces stable and tests per layer); scope growth (keep the Tkinter utility and vision work outside this milestone); user-facing latency becoming noticeable (Whisper + TAMEV + TTS on CPU) — measure and set an explicit latency budget.

---

## 8. Dependency Map

### 8.1 Phase dependencies

| Phase | Depends on | Unlocks |
|---|---|---|
| 0 — Safety & Build Foundation | — | everything |
| 1 — Core Architecture | 0 | 2, 3, 5, 6, 9, 10 |
| 2 — Tokenizer Foundation | 1 | 5, 8 |
| 3 — Config & Resolver | 1 | 4, 5 |
| 4 — AppManager | 3 | 5 |
| 5 — Action Executor | 1, 2, 3, 4 | 8, 12 |
| 6 — TAMEV C++ | 1 (verify against a fixed artifact) | 8 |
| 7 — Training/Data/Export | — (uses `ml_env`) | 8 (fine-tuned model) |
| 8 — Full Pipeline | 2, 5, 6, and 7 for the fine-tuned model | 12, 13, 15 |
| 9 — Input Automation | 1, 5 | `type_text` in 8, 15 |
| 10 — Screen Capture | 1 | 11, 12 |
| 11 — Vision / Screen Understanding | 10 | advanced `read_screen`, stronger verification |
| 12 — Verification | 5, 10 | 15 |
| 13 — TTS | 5, 8 | 15 |
| 14 — Reliability / Portability | 0 (plus fixes for code from 1–13) | 15 |
| 15 — Final Integration | 8, 9, 10, 12, 13, 14 | — |

### 8.2 Critical path

```
0 ─▶ 1 ─┬─▶ 2 ──────────────┐
        ├─▶ 3 ─▶ 4 ─────────┼─▶ 5 ─┬─▶ 8 ─▶ 15
        └─▶ 6 ──────────────┘      │
                 ▲                 ├─▶ 9 ──┘
   7 ────────────┘ (fine-tune)     ├─▶ 10 ─▶ 11
                                   ├─▶ 12 ──┘
                                   └─▶ 13 ──┘
   14 (hardening) ───────────────────────────▶ 15
```

**Minimum path to a Mira that does something useful:** `0 → 1 → 3 → 4 → 2 → 5` gives a typed-command application launcher with no ML. Adding `6` (TAMEV) and `7` (a good model + export) turns it into a real assistant. Everything from `9` onward makes it broader, not possible.

**Deliberately independent tracks** (can proceed in parallel once their inputs exist): `6`/`7` (ML), `9` (input), `10` (screen), `13` (TTS), and documentation/config work.

**Do not start** `11`, `12`, `15` before `5` and `8` are green — they have nothing to verify or integrate against.

---

## 9. ML / Dataset Strategy

Explicit constraint for this roadmap: **no dataset is generated, no model is trained, and no model is exported during planning.**

### 9.1 Intended workflow

```
Dataset ─▶ Train ─▶ Validation ─▶ Test ─▶ Error analysis ─▶ Improve dataset
      ─▶ Retrain ─▶ Export correct fine-tuned model ─▶ C++ ONNX inference
      ─▶ Python/C++ parity testing
```

Each arrow is a gate: do not proceed to "improve dataset" without error analysis, and do not proceed to "C++ inference" without export verification and parity.

### 9.2 Current ML facts (from the analysis)

- **Datasets:** `models/training/datasets/open_application.jsonl` (1165 rows) and `search_web.jsonl` (1016 rows) exist but are **not consumed** by `train.py`. Only the 56/7/7 split of the 70 hard-coded examples in `dataset.py` is used, which is the likely cause of the ~57 % accuracy.
- **Model:** TAMEV TinyBERT-style option-selection model — `models/tamev-base/` (14.39 M params, hidden 312, projection 64, temperature 1.34, `max_state_len` 128, `max_opt_len` 48) plus the `models/training/mira-tamev/` fine-tune.
- **Artifact:** `models/model_int8.onnx` is the **downloaded base** release, not an export of `mira-tamev`. No export script exists.
- **Contract:** logits/probs are positional over the candidate list, so option ordering is a hard interface between training and inference.

### 9.3 Strategy

1. **Fix plumbing before adding data.** Connect the existing 2181 examples, split them properly, and measure. Data volume is not the current bottleneck — the pipeline is.
2. **Quality over quantity.** When data *is* expanded, prioritise: diversity, natural-language variation, realistic commands, different word orders, short and long commands, casual speech, imperfect English, ambiguity, hard negatives, balanced intents, no duplicates, correct labels.
3. **Per-intent metrics are mandatory** (precision/recall/confusion matrix), because a single accuracy number hides which intents are unusable.
4. **Balance across all seven intents**, since the existing `datasets/` files cover only two of them; the other five intents currently rely on 10 examples each.
5. **Parity is a release gate.** A model enters the runtime only after C++/Python agreement on a frozen utterance set, comparing probability vectors, not just argmax.
6. **CPU-first.** Keep the quantised int8 ONNX path, bounded context lengths, and no GPU requirement; measure inference latency and keep it inside the end-to-end budget.
7. **Model revision is a versioned artifact:** record which dataset, hyper-parameters and export command produced each ONNX file so a regression can be traced and reverted.
8. **No new ML components without justification** (principle 9/10): intent classification earns its place; alias matching, path handling and token extraction do not.

---

## 10. Linux Portability Strategy

The development machine (GNOME + Wayland) is a *development environment*, not the architectural target. Two things are required for the environment: capability detection, and a backend interface per platform concern.

### 10.1 Session model

| Environment | Screen capture | Input injection |
|---|---|---|
| **Wayland + portal (GNOME, KDE, etc.)** | `org.freedesktop.portal.Screenshot` / Screencast via `libportal` (installed 0.11.0) | RemoteDesktop portal → EIS via `libei`/`liboeffis` (headers `/usr/include/libei-1.0`, version 1.6.0) |
| **Wayland + wlroots-style compositor** | `wlr-screencopy`-family protocols (external helper tools live here, e.g. `grim`) | `virtual-keyboard`/`wlr`-family protocols; `wtype` where available |
| **X11** | XCB + XRandR (installed) | XTest via `libXtst`/XCB (installed) |
| **External tools** | `import`/`ffmpeg` as optional fallbacks | `ydotool` (requires daemon + uinput permissions), `wtype` — **optional, never required** |
| **Headless / no session** | `Unavailable` (must be a valid, tested outcome) | `Unavailable` |

### 10.2 Rules

1. **Detect, do not assume.** Capability probes check environment variables (`XDG_SESSION_TYPE`, `WAYLAND_DISPLAY`, `DISPLAY`), the presence of a portal on the D-Bus session bus, and a backend self-test. `XDG_CURRENT_DESKTOP` may inform diagnostics but must never be the sole decision input.
2. **One file per backend.** Portal, Wayland/compositor-specific, and X11 code live in separate translation units implementing the same interface.
3. **GNOME-specific behaviour is quarantined.** If a GNOME workaround is unavoidable (for example a Shell D-Bus interface), it lives inside exactly one backend, is documented as such, and is never a dependency of the core pipeline. Today, `xdg-desktop-portal-gnome` is installed, which makes the portal path the primary route.
4. **`grim` is not a design dependency.** It is wlroots-only and does not work on GNOME Wayland despite being installed.
5. **No desktop environment may be required to build.** Backends are optional CMake features; a headless build must succeed and return `Unavailable` at runtime.
6. **Permissions are a normal outcome.** Screen and input access on portal-based systems may require a user-approved prompt on first use. Mira must surface this clearly and never hang or crash because of it.
7. **Known portability limits must be stated, not hidden.** Example: GNOME's portal exposes no window list or window title, so window enumeration is a *capability*, not an assumption, and features that need it must degrade cleanly.
8. **The build is the first portability boundary.** `CMakeLists.txt` currently makes PipeWire a `REQUIRED` dependency; audio should become optional/detected too, so Mira can be configured on machines with different audio stacks.

---

## 11. Testing Strategy

Testing is a deliverable of every phase, not a final phase. Tests must run without a microphone, a display, or a GPU wherever possible.

### 11.1 Test levels

| Level | Purpose | Examples | Runs in CI? |
|---|---|---|---|
| **Unit** | component behaviour in isolation | intent round-trips, text normalisation, tokenizer extraction tables, config parsing/validation, alias resolution, key mapping, `Result` propagation | yes |
| **Integration** | real collaborators, no hardware | tokenizer → resolver → AppManager **dry-run**; dispatcher with real resolver and a fixture `apps.json` | yes |
| **Model parity** | Python ↔ C++ agreement | same utterances through the Python reference and `TamevClassifier`; compare argmax **and** probability vectors; verify ONNX vs PyTorch max-abs-diff after export | yes (requires the model file; skip with a clear message if absent) |
| **End-to-end (text mode)** | whole pipeline without audio | intent × action matrix driven by typed text: happy path + failure path per intent | yes |
| **End-to-end (voice)** | the real interaction | hotkey → mic → Whisper → … → TTS on the development machine | no — manual checklist |
| **Regression** | previously broken things stay fixed | no busy-spin without a keyboard device; clean microphone start/stop/restart; low-confidence intent never triggers an action; `Unavailable` paths never crash | yes |

### 11.2 Rules

1. **Deterministic by default.** Anything requiring audio hardware, a display, a portal, or a network is a *manual* or *guarded* test, never a default CI test.
2. **No real launching in automated tests.** AppManager tests use `launch_dry_run`; at most one opt-in manual test performs a real launch.
3. **Injection points for anything non-deterministic:** resolver, AppManager, screen capture, input backend, TTS, and clock must be interface-backed so tests can substitute fakes.
4. **Fixtures over mocks where possible:** a temporary `apps.json`, a scratch directory with a dummy script/executable/`.desktop` file, and small synthetic images for OCR.
5. **Frozen golden sets for ML:** a committed utterance list with expected intents, versioned alongside the model, so a model change shows up as a diff.
6. **Skips are explicit.** A test that cannot run (no model, no display) reports a skip reason; it never silently passes.
7. **`ctest` is the single entry point** for all automated tests, with labels/tags per level so a headless run can select a subset.
8. **Sanitisers where practical** (`-fsanitize=address,undefined`) for the audio-thread and resolver/exec paths, given the data races already identified in `audio/microphone.cpp`.
9. **Manual verification checklist per phase**, recorded in the phase's completion notes (for example "recorded and transcribed a spoken 'open Firefox'").

---

## 12. Security / Safety Considerations

Mira executes real operations on the user's machine. The following rules are not optional.

### 12.1 Command execution

1. **No shell by default.** Execute via explicit `argv` (`posix_spawn`/`execve`). Never build a command string by concatenating user or model text.
2. **When a shell is genuinely required** (`method: script`), treat the script path as a configured absolute path from `apps.json` — never a path derived from speech.
3. **The spoken text never becomes the program name.** The program comes from user configuration; speech only selects *which* entry (a key into a trusted table).
4. **Reject, do not sanitise.** If a resolved value contains characters that would change the meaning of execution (shell metacharacters, embedded NUL, newlines), reject with a clear error instead of escaping and hoping.
5. **No fallback escalation.** If resolution fails, do not try "just run it anyway", and do not fall back to `xdg-open` on arbitrary input without an explicit, deliberate rule.

### 12.2 Application launching

1. **The allow-list is the configuration.** Only applications present in `apps.json` can be launched by name; anything else is "not configured".
2. **Validate before spawn:** existence, executability, `working_directory` validity and `.desktop` presence are checked first, so failures are reported rather than partially applied.
3. **Ambiguity is a refusal, not a coin flip.** Multiple matches → ask, never pick silently.
4. **Log every launch** (what, when, how, outcome) so a mistaken action is traceable afterwards.

### 12.3 File operations — highest-risk intent

1. **Treat `file_operation` as `FUTURE` and gated.** Do not enable it before a written policy exists.
2. The policy must include: an explicit allow-list of writable roots; **dry-run/confirmation for destructive verbs** (delete, overwrite, move across roots); no recursive deletion; no following symlinks out of the allowed roots; and refusal when the target is ambiguous ("this file" with no clear referent).
3. **Never act on a path inferred from speech without showing it back** to the user before a destructive operation.
4. Prefer trash-based deletion (`gio trash`-style) over unlink where available, with the choice documented.

### 12.4 Input injection

1. Typing and clicking are blind: they go to **whatever window is focused**. Never inject immediately after an action that may change focus; define an explicit focus/confirmation policy (for example: inject only after the user confirms the target application, and never type into a password field).
2. **Never type secrets.** There is no plausible reason for Mira to type passwords from speech; refuse, and document the refusal.
3. Key injection must not be able to synthesise dangerous chords accidentally; restrict the supported chord set to those actually needed.
4. Log a redacted record of injected actions (counts and target application, not full typed content).

### 12.5 Screen access

1. **Screenshots are sensitive.** Keep them in memory; if written to disk for debugging they must be opt-in, timestamped and documented as potentially containing private data.
2. Minimise capture: prefer regions/windows over full screens, and capture only when an intent actually needs it.
3. Never transmit captures anywhere — all processing is local, consistent with the CPU-local design.
4. State plainly when a capture is in progress (a visible indicator is preferable to a silent one).

### 12.6 Model and data handling

1. Whisper, TAMEV and any future model run **locally**; no audio, screen content or transcripts leave the machine.
2. Microphone capture is bounded (watchdog maximum duration) and always stoppable.
3. Logging must not become a covert recording: keep transcripts opt-in and redactable.

### 12.7 General

1. **Fail closed.** Uncertain intent or ambiguous entity → do nothing, and say so.
2. **Least privilege.** Do not request more permissions than the feature in use requires; portal-mediated access is preferred over broad group membership such as `input`, which grants far more than Mira needs.
3. **No privilege escalation.** Mira must not require root; `sudo`/`pkexec` are out of scope.
4. **Concurrency safety.** One action at a time by default; a second activation while an action is in flight is rejected or queued deliberately, never interleaved.
5. **Auditability.** Every action produces an `ActionRecord` that can be inspected afterwards.

---

## 13. Definition of Done

MiraV2 is "done" for the scope of this roadmap when all of the following hold.

### 13.1 Functional

- The full loop works by voice: activation chord → capture → transcript → intent → extraction → action → verification → spoken response.
- These intents work end to end with tests: `open_application`, `search_web`, `type_text`, `system_control`, `read_screen`, plus `conversation` routing.
- `open_application` works for all five launch methods (`command`, `executable`, `script`, `appimage`, `desktop`) from a user-edited `config/apps.json`.
- `file_operation` is enabled **only if** the §12.3 policy is implemented and tested; otherwise it remains explicitly unsupported.
- Mira reports failures, ambiguity and permission problems in spoken language, and never claims success without evidence.

### 13.2 Architectural

- `Intent` has exactly one definition in the codebase, and `tamev/option_catalog.hpp` is the only candidate list.
- TAMEV runs before the tokenizer, enforced by the API: extraction requires an intent argument.
- Tokenizer, resolver, AppManager, executor and backends each own exactly one responsibility, with no duplicated alias, normalisation or execution logic.
- `CommandManager` no longer exists, or exists only as a documented deprecated shim carrying no intent vocabulary.
- Every platform-specific capability is behind an interface selected by capability detection.
- No component depends on GNOME, Wayland, X11 or any external tool as a build-time or logical requirement.

### 13.3 Quality

- `ctest` covers unit, integration, parity, text-mode end-to-end and regression levels, and passes on a clean checkout.
- Python/C++ model parity holds on the frozen utterance set for the shipped model.
- Per-intent accuracy is measured and reported, and each enabled intent has an explicitly stated acceptable threshold.
- No known data races remain in the audio path; no leaked fds, threads or subprocesses across repeated start/stop cycles.
- A documented manual voice checklist passes on the development machine.

### 13.4 Operational

- The project is under version control with a documented build (`cmake` + `ctest`), a documented model-provisioning step, and a documented configuration file format.
- Models, configs and datasets are traceable: each ONNX artifact can be traced to the dataset and export command that produced it.
- The repository builds without network access, given the documented external assets.
- Mira starts on a machine that is missing optional backends and clearly reports what is unavailable.

### 13.5 Explicitly **not** required for "done"

Window enumeration, vision-based UI interaction, GUI configuration-utility polish, multi-language ASR, and verification beyond the documented best-effort level are all `FUTURE` and must not block completion.

---

## 14. Future Expansion

Ideas that are valuable but must not block the initial implementation. None of these may be pulled forward at the cost of the core loop.

### 14.1 Assistant capability

- **Continuous / wake-word mode** instead of a push-to-talk chord, including a local wake-word detector.
- **Conversation memory** and multi-turn disambiguation ("open it", "the other one") grounded in the `ActionRecord` history — a *separate* component from TAMEV, never a change to the intent classifier's contract.
- **Streaming ASR** with partial transcripts to cut perceived latency.
- **Non-English ASR and multilingual TAMEV** (the current pipeline is English-only: `params.language = "en"`, BERT-cased English vocabulary).
- **README/documentation polish** and a user-facing guide.

### 14.2 Environment breadth

- Additional screen-capture backends per compositor; a Wayland path that is not portal-mediated where such protocols exist.
- A window/application-state interface where the platform supports it (the current GNOME/Wayland limitation is documented in Phase 12).
- Multi-monitor and per-window capture policies; HiDPI-aware capture and injection.
- Audio backend abstraction beyond PipeWire (PulseAudio/ALSA), so the build is not tied to one audio stack.

### 14.3 Intelligence

- Local LLM for `conversation` and for paraphrasing responses (kept strictly behind an interface, CPU-budgeted, and never in the intent-decision path).
- Vision models for UI element detection and app-state recognition (Phase 11 extension).
- Learning/improvement loop from `ActionRecord` history to propose new aliases or dataset entries for review (human-in-the-loop, never automatic training).

### 14.4 Ecosystem

- The Tkinter/Python `apps.json` editor, extended into a small "check my setup" wizard that validates models, backends and permissions.
- Import/export of application configurations and a shareable community config format.
- Additional action types: clipboard operations, window management, notifications, calendar/media control.
- Packaging (AUR/deb/flatpak) with the models handled as external assets.
- A plugin/action SDK so third parties can add intents without touching core — contingent on the interfaces in Phases 1–5 being stable first.

### 14.5 Explicitly rejected directions (unless requirements change)

- Making Mira depend on GNOME-specific APIs as a *requirement* rather than an optional backend.
- Any cloud service for ASR, intent classification, vision or TTS — the design is deliberately local and CPU-first.
- Replacing the deterministic resolver with an LLM ("just ask a model which app to open"), which trades verifiability for novelty.

---

*End of roadmap. No implementation is authorised by this document; each phase requires explicit approval before work begins.*

