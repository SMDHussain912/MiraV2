# MiraV2

A modular, local-first desktop assistant for Linux, written in C++.

> **Project status:** early development. Speech recognition, audio capture, the activation hotkey and a legacy command splitter are implemented; the intent-routing, resolution and execution layers described below are planned or only partially implemented. See [Current Status](#13-current-status) and [ROADMAP.md](ROADMAP.md).

---

## 1. Overview

MiraV2 is an experimental, local-first computer assistant project for Linux. The goal is an assistant that performs real desktop operations — launching applications, searching the web, typing text, manipulating files, controlling the system, reading the screen — rather than only returning generated text.

The design deliberately splits the work across specialised components instead of relying on a single model for every task: speech recognition, intent decision/routing, intent-aware extraction, resource resolution, execution, computer interaction, screen understanding, verification and response generation are separate stages with explicit responsibilities.

Processing is intended to run locally: speech recognition and intent routing are designed to work on the CPU using small, quantised models, with no cloud service required for the core loop.

Current state at a glance:

- **Implemented:** PipeWire audio capture, Whisper speech recognition over `whisper.cpp`, an evdev activation-hotkey listener, a legacy first-word command splitter, and an intent-aware tokenizer covering one intent.
- **Not implemented:** intent routing inside the C++ runtime, application resolution and launching, the action executor, input automation, screen capture, vision/OCR, verification and TTS.

The complete phased plan is in [ROADMAP.md](ROADMAP.md).

---

## 2. Architecture

The intended runtime flow. The ordering of the first three stages is an architectural rule, not an implementation detail: **TAMEV decides the intent first, and the tokenizer extracts information only after the intent is known.**

```
User speech
   │
   ▼
Speech Recognition (Whisper)
   │  transcript
   ▼
TAMEV ──────────────────► decides the intent (what kind of operation is being requested)
   │  Intent
   ▼
Tokenizer ──────────────► extracts the target / query / text / path for that intent
   │  extracted fields
   ▼
Resolver ───────────────► locates the configured resource (e.g. an application entry)
   │  resolved specification
   ▼
Executor ───────────────► dispatches to the tool responsible for the intent
   │
   ▼
Tools / platform backends
   ├── Application manager   (launch a configured application)
   ├── Input automation      (typing, key presses, mouse)
   └── Screen reader ──► Vision / screen understanding    [only when an intent requires it]
   │
   ▼
Verification ───────────► did the action actually succeed?
   │
   ▼
Response / TTS
```

Screen capture and vision are optional participants: they are involved only for intents that need to observe the desktop, and they also feed the verification stage.

Stage responsibilities:

| Stage | Responsibility |
|---|---|
| Speech recognition | Convert captured audio into text. |
| TAMEV | Decide the intent: which kind of operation the user is asking for. |
| Tokenizer | Given the intent, extract the relevant value (application reference, search query, text to type, path, control command). |
| Resolver | Turn an extracted reference into a concrete, configured resource (for example an application launch specification). |
| Executor | Route the intent and the extracted fields to the correct action. |
| Tools / backends | Perform the operation (launch, type, click, capture). |
| Verification | Determine whether the operation succeeded, and report honestly when it cannot be determined. |
| Response / TTS | Produce the spoken or textual result. |

---

## 3. Design Principles

These principles are binding for the project and are documented in more detail in [ROADMAP.md](ROADMAP.md).

1. **Modular architecture.** Each stage has one responsibility and is replaceable.
2. **Local-first processing.** Speech recognition and intent routing are intended to run on the local machine, with no cloud dependency for the core loop.
3. **CPU-oriented design.** Small models and quantised inference; no GPU requirement.
4. **Deterministic tools where appropriate.** Alias matching, path handling and extraction are ordinary deterministic code, not model calls.
5. **ML only where it provides meaningful value.** Open-ended natural-language intent classification justifies a model; string and filesystem work does not.
6. **Clear separation between decision, extraction, resolution, execution and verification.** TAMEV decides, the tokenizer extracts, the resolver finds resources, the manager/executor performs the operation, verification checks the result.
7. **Platform abstraction.** Platform-specific code lives behind backends selected by capability detection.
8. **Linux portability.** Designed for portability across Linux desktop and session environments; no single desktop environment is the architectural target.
9. **Security and fail-closed behaviour.** When intent or the target is uncertain, Mira does not act.
10. **Testability.** Components must be testable without a microphone, display or classifier.
11. **Extensibility.** New intents and new platform backends should be additive, not invasive.

---

## 4. Core Components

The status column reflects the actual state of the repository, not the design intent.

| Component | Responsibility | Status |
|---|---|---|
| **Speech recognition (Whisper)** | Convert captured audio into text using `whisper.cpp`. | Implemented (`speech/`) |
| **Audio capture** | Capture microphone audio as 16 kHz mono float for Whisper, via PipeWire. | Implemented (`audio/`); lifecycle and device-selection limitations are tracked in the roadmap |
| **Activation hotkey** | Wait for the user's activation chord before recording. | Implemented (`input/`); currently bound to one hardcoded input device on the development machine |
| **Tokenizer** | Intent-aware extraction of the target/query/text/path for a known intent. | Partial (`tokenizer/`); implemented for `open_application` only; wired into the runtime through `pipeline/` (Phase 8) |
| **TAMEV** | Decision/routing layer that selects the user's intent. | Partial: model definition, configuration and Python training/evaluation tooling are present in `models/` (weights are not tracked); C++ inference integration is planned |
| **Resolver** | Turn an extracted reference into a concrete configured resource and produce a launch specification. | Planned |
| **Application manager** | Launch applications from a resolved specification using the configured launch method. | Placeholder (`appmgr/appmgr.{hpp,cpp}` are empty); planned |
| **Executor** | Route `Intent + extracted fields` to the tool that performs the operation. | Planned |
| **Input automation** | Type text, press keys, use hotkeys, move and click the mouse, behind a backend interface. | Planned |
| **Screen reader** | Capture screen content through backend-specific mechanisms with capability detection. | Placeholder (`screen_reader/SReader.{hpp,cpp}` are empty); planned |
| **Vision / screen understanding** | Interpret captured images (OCR, UI elements, application state) when an intent requires it. | Planned |
| **Verification** | Check whether an action actually succeeded, with an explicit "unverified" outcome. | Planned |
| **TTS** | Speak responses through a replaceable backend. | Planned |

---

## 5. TAMEV

TAMEV is MiraV2's **decision/routing layer**. Its only job is to determine what kind of operation the user is requesting: it receives the transcribed request together with a fixed list of candidate actions, and selects one of them.

It is **not** a general-purpose conversational language model, and it is **not** a vision system. It performs option selection over a closed set of intents.

Intent set currently defined by the project's training configuration:

```
open_application    search_web    read_screen    type_text
file_operation      system_control    conversation
```

How it works, as reflected by the repository:

- **Architecture:** a small TinyBERT-style encoder with a bilinear "pointer" head that scores the context against each candidate option and returns a probability distribution over them. The base model is approximately 14.4 M parameters (`hidden_dim` 312, `projection_dim` 64, tier `nano`), with a limited context length, which keeps it CPU-friendly.
- **Interface:** context + candidate options in, a distribution over the options out. The output is positional: the order of the candidate list is a contract, so the C++ side must use exactly the same ordering as training.
- **Artifacts present locally (weights are not tracked in git):** the base model definition and configuration under `models/tamev-base/`, a locally fine-tuned model under `models/training/mira-tamev/`, and an int8 ONNX export used by the Python tooling for CPU inference.
- **Python tooling present:** `models/tester.py` (runs the ONNX model), `models/runners/tester.py`, and `models/training/` (`train.py`, `dataset.py`, `split_data.py`, plus small train/validation/test JSONL files and larger per-intent dataset files).

Current limitations, stated plainly:

- **C++ inference is not implemented.** TAMEV is not part of the running runtime today; the runtime still uses the legacy command splitter.
- **The ONNX file in the working tree is the released base artifact**, not an export of the local fine-tune. A reproducible fine-tune → ONNX export step does not exist yet.
- **Intent accuracy is not yet acceptable**, and the larger generated datasets in `models/training/datasets/` are not yet consumed by the training script.
- **Confidence gating is planned:** below a threshold, the intent should resolve to "unknown" and Mira should ask instead of acting.

Model weights are intentionally not stored in this repository (see [Repository Structure](#12-repository-structure)).

---

## 6. Tokenization and Resolution

MiraV2 separates four things that are often conflated. The distinction is deliberate:

| Stage | Question it answers |
|---|---|
| **TAMEV** | *What kind of operation is being requested?* |
| **Tokenizer** | *What is the relevant value inside this request, given that intent?* |
| **Resolver** | *Which concrete configured resource does that value refer to?* |
| **Executor / manager** | *Perform the operation.* |

The order matters: the tokenizer runs **after** TAMEV and never guesses intent. It is not a general-purpose NLP component — it extracts the field the selected intent needs.

Example, `open_application`:

```
user says        "Open Firefox"
Whisper          "Open Firefox"
TAMEV            intent = open_application
Tokenizer        TARGET = "Firefox"            (command verb removed, value extracted)
Resolver         "Firefox" -> configured launch specification
Application mgr  launches the application
```

Example, `search_web`:

```
user says        "Search the internet for how transformers work"
TAMEV            intent = search_web
Tokenizer        QUERY = "how transformers work"
Executor         performs the search
```

The current tokenizer implementation covers `open_application` only (recognising `open`, `launch`, `start` and `run` prefixes, case-insensitively, and returning the remaining application reference). Extraction for the other intents is planned. The resolver is planned and does not exist yet; in particular, the resolver is not permitted to launch applications — resolution and execution are separate responsibilities.

---

## 7. Linux Platform Architecture

MiraV2 is designed for portability across Linux desktop and session environments. It is a Linux project: there is no Windows or macOS support.

The development machine currently used for the project is a GNOME/Wayland session with PipeWire, but **that environment is not the architectural target**. The design intent is that platform differences are handled by interchangeable backends behind common interfaces, selected by capability detection rather than by assuming a particular desktop.

Intended coverage:

| Environment | Screen access | Input automation |
|---|---|---|
| Wayland with `xdg-desktop-portal` (GNOME, KDE, and others) | Screenshot / screencast through the portal interface | RemoteDesktop portal + EIS input-emulation interfaces |
| wlroots-style compositors | Compositor-specific screencopy-style protocols | Compositor-specific input protocols |
| X11 | XCB / XRandR-based capture | X11 input-synthesis mechanisms |
| External helper tools | Optional fallback only | Optional fallback only (never a hard dependency) |
| No session / headless | Reported as "unavailable" | Reported as "unavailable" |

Consequences of this approach:

- Capability detection is preferred over assumptions; the current desktop is used for diagnostics, not as the deciding factor.
- Frontend and platform concerns stay separate: each backend is a separate implementation of one interface.
- Restricting the desktop environment may only mean a capability reports "unavailable" — it must not mean a crash or an obscure failure.
- Interactive permission prompts (for example, screen access or input emulation on portal-based systems) are treated as normal outcomes that must be reported clearly.

None of these backends are implemented yet. Screen capture, screen understanding and input automation are planned; see [ROADMAP.md](ROADMAP.md).

---

## 8. Application Management

Applications cannot be assumed to be launchable through one mechanism on Linux, so resolution is configuration-driven. The intended model is a user-editable JSON file (for example `config/apps.json`) describing the applications Mira knows about:

```json
{
    "version": 1,
    "applications": [
        {
            "name": "Firefox",
            "aliases": ["firefox", "mozilla firefox"],
            "launch": {
                "method": "command",
                "command": "firefox",
                "working_directory": null
            }
        }
    ]
}
```

Planned launch methods:

| Method | Meaning |
|---|---|
| `command` | A program resolved through `PATH`, with optional arguments. |
| `executable` | An absolute path to an executable. |
| `script` | A script at an absolute path. |
| `appimage` | An AppImage file. |
| `desktop` | A `.desktop` entry resolved through the standard XDG data directories. |

Division of responsibility:

- **Resolver (planned).** Loads and validates the configuration, normalises aliases and names, and turns an extracted application reference into a concrete launch specification. Resolution performs no side effects.
- **Application manager (planned).** Takes a resolved specification and performs the launch. It does not search for applications, read the configuration, or decide whether an application should be launched.
- **Executor (planned).** Connects an `open_application` intent and its extracted target to the resolver and then the manager.

**Status: not implemented.** The `appmgr/` files exist but are empty placeholders, the resolver does not exist, and no configuration file is present yet. MiraV2 cannot launch applications today. The design constraints that are already fixed: execution is performed with an explicit argument vector rather than by concatenating text into a shell command, only configured (allow-listed) applications can be launched by name, and an ambiguous match must be refused rather than guessed.

---

## 9. Vision and Screen Understanding

Three distinct layers are involved, and they are deliberately kept separate:

| Layer | What it is | Status |
|---|---|---|
| **Screen capture / access** | Obtaining pixels (full screen, region, or window where the platform allows) through a platform backend, with capability detection. | Planned (`screen_reader/` files are empty placeholders) |
| **Computer-vision toolkit** | General image processing — scaling, cropping, region selection, thresholding, change detection — as supplied by a toolkit such as OpenCV. | Planned; OpenCV is not part of the current build |
| **Screen / UI understanding** | Interpreting the image: OCR text with regions, candidate UI elements, application/window state. This is where specialised vision models may eventually be used. | Future work |

Important distinctions:

- **OpenCV is a computer-vision toolkit, not an AI vision model.** It provides the pixel-level operations that vision work is built on; it does not by itself understand a user interface.
- **Capture and interpretation are separate concerns.** Screen capture must work without any vision model being present, and a vision model must be replaceable without changing how pixels are obtained.
- **No OCR engine is currently selected or integrated.** Nothing in the repository performs OCR today.
- Screen access is treated as sensitive: captures are intended to stay in memory, only be taken when an intent requires them, and never be sent anywhere.

Vision is intended to help Mira understand visual UI state when a task requires it — for example, to answer "what does this say?" or to support verification that an action had the intended visible effect. It is not required for the core launch-an-application flow.

---

## 10. Security

MiraV2 performs real operations on the user's machine, so security constraints are part of the architecture rather than an afterthought. These constraints are established in the project's design; their enforcement is tied to the corresponding implementation phases, and the relevant components are not built yet.

**Execution**

- No unrestricted arbitrary shell execution derived from model or speech output. Operations are performed with an explicit argument vector, not by building a shell command string.
- The executable is never taken from spoken text: speech selects an entry from the user's own configuration, and the program comes from that configuration.
- When a shell is genuinely required, only a configured absolute path is used.
- Invalid or ambiguous values are rejected rather than escaped and retried, and a failed resolution is not "retried anyway" through an escalating fallback.

**Controlled application launching**

- Only applications present in the configuration can be launched by name.
- Resolution is validated before anything is launched (existence, executability, working directory).
- An ambiguous match is refused and surfaced to the user rather than resolved by guessing.

**Fail-closed behaviour**

- Uncertain intent, low classifier confidence, or an ambiguous target results in no action, plus a clear message.
- A failure is never converted into a different action the user did not request.

**Sensitive operations**

- Destructive file operations are gated behind an explicit policy (allow-listed roots, confirmation for destructive verbs) and are not enabled until that policy exists.
- Input automation never types secrets on the user's behalf, and is subject to a focus policy because synthetic input goes to whatever window is focused.
- Screen captures are treated as sensitive data.

**Privileges and locality**

- MiraV2 does not require root and performs no privilege escalation.
- Permissions are requested narrowly, per feature, in preference to broad group memberships.
- Speech recognition and intent classification run locally; the core loop has no cloud dependency.
- Screenshot and transcript data are not transmitted anywhere.

**Separation of decision and execution**

- The model decides *what* is requested; separately implemented, deterministic code decides *whether and how* it is performed.

**Status:** none of the execution, resolution, file-operation or input-automation components are implemented yet, so the majority of these constraints are not yet exercised by code. They are recorded in [ROADMAP.md](ROADMAP.md) §12 as requirements for the phases that introduce them.

---

## 11. Technology Stack

**Used in the project today**

| Technology | Role |
|---|---|
| **C++ (C++17)** | Implementation language; the standard is set in `CMakeLists.txt`. |
| **CMake** (≥ 3.16) + pkg-config | Build system; `libpipewire-0.3` is currently a required dependency at configure time. |
| **PipeWire** | Microphone capture (`audio/`). |
| **whisper.cpp** | Speech recognition. Fetched and built from source at a pinned revision as part of the MiraV2 build, and linked statically into the executable (no runtime library to install). The GGML `base.en` model is provisioned locally and is not tracked in the repository. |
| **Linux evdev** (`linux/input.h`) | Reading the activation hotkey from an input device. |
| **Python 3** | Modelling, training and evaluation tooling under `models/`. |
| **PyTorch + Transformers** | TAMEV model definition, fine-tuning and evaluation scripts. |
| **ONNX / ONNX Runtime (Python)** | Int8 export and Python-side inference of the intent model. |
| **scikit-learn** | Dataset splitting in the training tooling. |
| **WordPiece tokenizer assets** (`tokenizer.json`) | Text encoding for the TAMEV model (BERT-style, ~30 k vocabulary, lowercasing). |

**Established in the planned architecture, not yet used**

| Technology | Intended role |
|---|---|
| **ONNX Runtime C/C++ API** | In-process TAMEV inference. |
| **nlohmann/json** | Parsing the application configuration and other JSON assets. |
| **OpenCV** | Image-processing toolkit for screen understanding. |
| **xdg-desktop-portal (libportal)** | Portal-based screen capture and input emulation access. |
| **libei / liboeffis** | Input emulation on Wayland via the RemoteDesktop portal. |
| **Wayland, X11 / XCB / XRandR** | Platform backends for capture and input. |
| **D-Bus** | Portal and desktop integration. |
| **A TTS engine** | Spoken responses (no engine has been selected yet). |

### Building

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build        # run the test suite
```

The build type defaults to `RelWithDebInfo` when none is specified, and the test suite is built by default (`-DMIRA_BUILD_TESTS=OFF` disables it).

Requirements: a C++17 compiler, CMake ≥ 3.16, `pkg-config`, and the `libpipewire-0.3` development headers.

whisper.cpp is not shipped as a binary: it is fetched and built from source at a pinned revision as part of the MiraV2 build, so the first configure needs network access to its git repository. The resulting library is linked statically into `miraV2`, so there is no shared library to install or resolve at runtime. To build without network access, point the build at an existing checkout instead of cloning:

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_WHISPER=/path/to/whisper.cpp
```

**Model weights are not stored in the repository** (see below), so a fresh clone cannot run the speech pipeline until they are provisioned locally:

- `models/ggml-base.en.bin` — speech recognition model, loaded by the current runtime from a path relative to the working directory, so run Mira from the project root.
- `models/model_int8.onnx` — quantised intent model, used by the Python tooling today and intended for the C++ runtime once implemented.

### License

No license file is currently included in this repository.

---

## 12. Repository Structure

```
miraV2/
├── CMakeLists.txt          Build configuration for the single `miraV2` target
├── cmake/                  Dependency configuration (whisper.cmake: pinned whisper.cpp source build)
├── ROADMAP.md              Full architecture and phased development plan
├── .gitignore
├── tests/                  Test suite, built and run through ctest
├── main.cpp                Current entry point: hotkey -> capture -> Whisper -> legacy splitter -> stdout
│
├── appmgr/                 PLACEHOLDER — empty files; planned application launching
├── screen_reader/          PLACEHOLDER — empty files; planned screen-capture backends
│
├── audio/                  PipeWire microphone capture
│   ├── microphone.hpp
│   └── microphone.cpp
├── speech/                 Whisper integration
│   ├── whisper.hpp
│   ├── whisper.cpp
│   └── whisper_lib/        Snapshot of the whisper.cpp public headers used by the wrapper
├── input/                  Activation-hotkey listener (Linux evdev)
│   ├── hotkey.hpp
│   └── hotkey.cpp
├── pipeline/               Full runtime pipeline (Phase 8)
│   ├── pipeline.hpp
│   └── pipeline.cpp
├── tokenizer/              Intent-aware extraction
│   ├── tokenizer.hpp
│   ├── tokenizer.cpp
│   └── test_tokenizer.cpp  Standalone test program (not yet part of the CMake build)
│
├── models/                 Intent model assets and modelling tooling
│   ├── tamev-base/         Base model definition and configuration
│   ├── training/           Training/splitting scripts and JSONL datasets
│   ├── runners/            Standalone model runner
│   └── tester.py           Python inference/evaluation of the ONNX model
│
└── build/                  Local build output (not tracked)
```

Only the following are compiled into the `miraV2` target itself today: `main.cpp`, `speech/whisper.cpp`, `audio/microphone.cpp`, `input/hotkey.cpp` — everything else (`pipeline/`, `tokenizer/`, `tamev/`, `resolver/`, `executor/`, `appmgr/`, `core/`) comes from the shared `mira_core` library. `screen_reader/` contains no implementation.

Directories described in the roadmap but **not present in the repository**: `vision/`, `verification/` and `tts/`.

### Model weights

Large model artifacts are excluded from version control via `.gitignore` (`*.bin`, `*.onnx`, `*.safetensors`, `*.gguf`, and model download caches). They exist only in local working copies and must be provisioned separately. Implementation code, scripts, small datasets and tokenizer configuration are tracked.

---

## 13. Current Status

Every item below is based on the state of the source tree, not on the roadmap's intentions. A component is listed as implemented only where working code exists.

### Implemented

| Item | Notes |
|---|---|
| Microphone capture (`audio/`) | PipeWire capture at 16 kHz mono float, suitable as Whisper input. Capture is a fixed-length window and the lifecycle has known limitations that are tracked for hardening. |
| Speech recognition (`speech/`) | Thin C++ wrapper over `whisper.cpp`: loads a GGML model and returns the concatenated transcript. Language, thread count and model path are currently fixed in code. |
| Activation hotkey (`input/`) | Reads key events from a Linux input device and waits for the activation chord. The device path is hardcoded, which makes this development-machine specific. |
| Intent-aware tokenizer (`tokenizer/`) | Extraction for `open_application`: recognises `open`, `launch`, `start`, `run` (case-insensitive) and returns the remaining application reference, with trimming and validity checking. Wired into the runtime through `pipeline/` (Phase 8). |
| Intent model assets and tooling (`models/`) | TAMEV base model definition and configuration, a locally fine-tuned model, an int8 ONNX export, plus Python scripts for inference/evaluation and training-data handling. Model weights are not tracked in the repository. |

**What actually runs today:** MiraV2 is a single executable that waits for the activation hotkey, records a short fixed-length clip, transcribes it with Whisper, splits the transcript with the legacy splitter, prints the result to stdout, and exits. **No action is performed** — no application is launched, nothing is typed, nothing is captured from the screen.

### In Development

- **Tokenizer coverage and integration.** Extraction currently exists for one intent; adding the remaining intents and wiring the tokenizer into the build and pipeline is the active workstream.
- **Intent-model workflow.** The training script currently consumes only a small dataset while larger per-intent dataset files exist unused, and there is no reproducible fine-tune → ONNX export step. Correcting this, and validating model output against the Python reference, is required before the model is used at runtime.
- **Foundation work (immediately next per the roadmap).** Build/test foundations, a single canonical intent representation shared by all components, and shared result/error types.

### Planned

- **TAMEV C++ inference** (ONNX Runtime) plus the WordPiece text encoder it requires, with Python/C++ output parity checks.
- **Resolver** with a versioned, user-editable application configuration (aliases, launch specifications).
- **Application manager** implementing the `command`, `executable`, `script`, `appimage` and `desktop` launch methods, with a dry-run mode.
- **Action executor** connecting intents and extracted fields to concrete actions.
- **Input automation** behind a backend interface (Wayland/portal and X11 paths, optional external tools).
- **Screen capture backends** with capability detection.
- **Vision and screen understanding** (image processing, OCR, UI element recognition).
- **Verification** of actions, including an explicit "unverified" outcome.
- **TTS** through a replaceable backend.
- **Reliability and portability hardening:** remove the hardcoded input device, fix the audio capture lifecycle, make paths independent of the working directory, add configuration and logging.
- **Automated test suite** (unit, integration, model parity, end-to-end and regression tests) exposed through `ctest`.

---

## 14. Development Roadmap

The complete development plan is maintained separately in **[ROADMAP.md](ROADMAP.md)**.

It covers the phased plan (Phase 0 through Phase 15), the dependency map between phases, component responsibilities, the data flow and failure paths, the ML and dataset strategy, the Linux portability strategy, the testing strategy, the security and safety requirements, the definition of done, and future expansion ideas.

This README intentionally summarises that document rather than duplicating it.

---

## 15. Development Status and Project Maturity

MiraV2 is under active development and should be treated as an experimental, work-in-progress project.

- The architecture and the interfaces between components are still being established and may change.
- Most of the described system does not exist yet; the sections above distinguish implemented, in-development and planned work, and that distinction should be taken literally.
- MiraV2 is not a finished or production-ready assistant, and no stability or compatibility guarantees are offered.
- The project is intended to be run locally by its developer and anyone willing to work with a moving target. There is no release, package or supported configuration at this time.
- The repository is public, but no license has been declared, so no usage rights are granted beyond viewing the code.

---

## 16. MiraV1

MiraV1 was the earlier, experimental prototype of this project. It was developed and used on the developer's local system for early exploration: capturing speech, transcribing it, and applying simple command handling to see whether a local voice-driven assistant was practical.

It was never a distribution: MiraV1 existed as a set of standalone experimental programs rather than a project with defined component boundaries, it was not published as a cross-platform release, and it carried no compatibility or maintenance guarantees. Its code is not part of this repository's supported codebase.

The issues that surfaced during that work — unclear boundaries between parsing, decision-making and execution; environment-specific assumptions; limited testability; and safety concerns around executing operations derived from speech — directly motivated the redesign.

MiraV2 is the architectural successor, redesigned around:

- **Modularity** — specialised components with single responsibilities instead of one program doing everything.
- **Clearer component boundaries** — an explicit pipeline from speech to response, with decision, extraction, resolution, execution and verification kept separate.
- **Linux portability** — platform behaviour behind interchangeable backends, rather than assumptions about one desktop environment.
- **Testability** — components that can be exercised without a microphone, a display or a classifier.
- **Security** — controlled execution, allow-listed launching, fail-closed behaviour on ambiguity, and no unnecessary privilege escalation.
- **Extensibility** — new intents and new platform backends added without restructuring the existing ones.
- **Local-first processing** — small CPU-oriented models running on the user's machine, with no cloud dependency for the core loop.

