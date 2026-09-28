# TAMEV Intent Classifier Datasheet

This specification describes the repository's current TAMEV task and the
requirements that follow from its interfaces and Phase 7 roadmap. It does not
generate examples, authorize training, or claim that the current checkpoint is
acceptable.

## Model Identity and Task

- **Mira model:** the fine-tuned PyTorch checkpoint at
  `models/training/mira-tamev/` (`model.safetensors`, TAMEV config, model code,
  and tokenizer files are present).
- **Initialization:** `models/tamev-base/`, whose config identifies a TAMEV
  nano encoder with TinyBERT `huawei-noah/TinyBERT_General_4L_312D`, hidden
  dimension 312, projection dimension 64, CLS pooling, and a bilinear pointer
  decision head. The config reports approximately 14.39 million parameters.
- **ONNX artifact:** `models/model_int8.onnx` is documented by the parity notes
  and roadmap as the downloaded released/base artifact, not an export of the
  local Mira fine-tune. Frozen vectors for it verify inference parity, not Mira
  task quality.
- **Purpose:** select one candidate intent for a user request. TAMEV is an
  option selector/classifier; it does not generate response text, extract the
  target/query, resolve resources, or execute an action.
- **Architecture order:** Whisper transcript -> TAMEV intent -> intent-aware
  Tokenizer -> ActionRequest -> ActionDispatcher -> action handler ->
  resolver/tool -> verification. TAMEV decides; extraction is downstream.

The training representation in `train.py` is a text request plus one label.
The context is `Which action should Mira perform?\nUser request: ` followed by
the request. The candidate options are supplied in a fixed positional order.
The model returns logits/probabilities over these candidates and their
argmax-selected option; it does not emit free-form text.

Canonical runtime inference uses a 128-token context and seven options padded
to 64 tokens (`tamev/option_catalog.hpp`, `tamev/tamev_classifier.hpp`, and
`models/parity/README.md`). The base config says `max_opt_len: 48`, while the
Python training config and runtime contract use maximum/padded option length
64. Record and resolve this configuration discrepancy before treating an
export as final. `models/tester.py` and `models/runners/tester.py` use a
different question/context and only four options (option length 32); they are
not the canonical Mira seven-intent evaluation interface.

## Canonical Intents

The exact order comes from `core/intent.cpp` / `intent_catalog()` and is also
present in the Python training and metrics lists. `Intent::Unknown` is a runtime
abstention/error state, not a model option or training class.

| Index | Exact label | Intended meaning in the current project |
|---:|---|---|
| 0 | `open_application` | Request to launch/open a desktop application. The roadmap example is “Open Firefox.” It is distinct from opening a file or folder. |
| 1 | `search_web` | Request to search the web for information. The roadmap example asks to search the internet for a topic. It is distinct from searching local files. |
| 2 | `read_screen` | Request to inspect/read what is displayed on the screen or window. It does not mean general conversation. Screen capture/vision capability is not implied by the label. |
| 3 | `type_text` | Request to enter/type text through input automation. The roadmap example is “Type hello world.” Input automation is a separate capability and safety boundary. |
| 4 | `file_operation` | Request to manage a local file or folder. The roadmap example is moving a file to Documents. File-operation safety policy remains a separate requirement. |
| 5 | `system_control` | Request to control an operating-system/device setting or state. The roadmap example is turning volume down. The specific supported control verbs are not comprehensively defined or implemented. |
| 6 | `conversation` | Requests routed as conversation rather than one of the explicit operation intents. Phase 8 describes this as routing only; no response-generating model is specified here. |

These meanings are the project's stated intent taxonomy, not proof that each
operation currently has an implemented handler. The single-label dataset
schema has no `Unknown`, `ambiguous`, or OOD label. Do not append such a label
without changing the model option contract and runtime mapping.

## Labeling Rules and Boundary Examples

Examples below are verbatim requests already present in `models/training/`
JSONL files or `dataset.py`. The “negative” example is a request explicitly
annotated with a different existing intent; it is useful as a boundary pair,
not a claim that the full negative distribution is represented. Ambiguous
examples are existing rows that expose unresolved labeling or execution
boundaries. They must not be copied as unquestioned gold labels without review.

### `open_application`

- **Belongs:** request to open/launch an application.
- **Does not belong:** opening a local folder/file (`file_operation`) or
  searching online (`search_web`).
- **Positive:** “Open Firefox” (`dataset.py`).
- **Negative:** “Open my Documents folder” (`dataset.py`,
  `file_operation`).
- **Ambiguous/review:** “Open Spotify and play something calm” and “Search for
  the app store and launch it” (`hard_negatives_v2.jsonl`, both labeled
  `open_application`). These combine steps; the dataset has no documented
  multi-intent policy. Do not infer that TAMEV can decompose them.
- **Out of domain:** no separate OOD set or OOD label is present.

### `search_web`

- **Belongs:** explicit web/internet/online-service search.
- **Does not belong:** searching the local computer or its folders
  (`file_operation`).
- **Positive:** “Search the web for Linux commands” (`dataset.py`).
- **Negative:** “Find this file on my computer” (`dataset.py`,
  `file_operation`).
- **Ambiguous/review:** “Can you search the internet for this?” (`dataset.py`).
  The intent words are clear but the query is unspecified; query extraction and
  clarification are downstream decisions, not TAMEV labels.
- **Out of domain:** no separate OOD set or OOD label is present.

### `read_screen`

- **Belongs:** a request explicitly about reading/inspecting displayed screen
  or window content.
- **Does not belong:** general questions about a subject that do not refer to
  the screen (`conversation`), or web searches (`search_web`).
- **Positive:** “Read the text on the screen” (`dataset.py`).
- **Negative:** “What do you think about programming?” (`dataset.py`,
  `conversation`).
- **Ambiguous/review:** “What application is currently open?” (`dataset.py`).
  It could require screen interpretation or window enumeration. The roadmap
  specifically notes that GNOME's portal does not expose a window list; the
  intent label alone cannot guarantee the needed evidence/capability.
- **Out of domain:** no separate OOD set or OOD label is present.

### `type_text`

- **Belongs:** request to type/enter/paste text into a field, editor, terminal,
  or current focus.
- **Does not belong:** request to compose/generate text without asking Mira to
  enter it (`conversation` in the current annotations). No generation model is
  defined by this task.
- **Positive:** “Type hello world” (`dataset.py`).
- **Negative:** “Write a haiku about rain” (`type_text_v2.jsonl`, labeled
  `conversation`).
- **Ambiguous/review:** “Type my name” (`dataset.py`). It identifies an input
  action but omits the text to enter. Missing content is not a different intent;
  how to clarify it is not specified here.
- **Out of domain:** no separate OOD set or OOD label is present.

### `file_operation`

- **Belongs:** request to manage a local file/folder (the seed examples include
  create, delete, move, rename, copy, show, or find files).
- **Does not belong:** changing device/system state (`system_control`) or
  searching online (`search_web`).
- **Positive:** “Move this file to Documents” (`dataset.py`).
- **Negative:** “Turn off the computer” (`dataset.py`, `system_control`).
- **Ambiguous/review:** “Delete this file” (`dataset.py`) lacks an explicit
  referent and is destructive. The intent can be `file_operation` while target
  resolution and authorization remain unresolved; do not treat the label as
  permission to act.
- **Out of domain:** no separate OOD set or OOD label is present.

### `system_control`

- **Belongs:** control of system/device state; seed examples include power,
  restart, lock, volume, and brightness.
- **Does not belong:** launching an application (`open_application`) or a
  general conversational request (`conversation`).
- **Positive:** “Decrease the volume” (`dataset.py`).
- **Negative:** “Start Blender” (`dataset.py`, `open_application`).
- **Ambiguous/review:** “Wake the machine up” (`system_control_v2.jsonl`) is
  present as a label, but the repository does not define which wake behavior
  Mira can cause or support. Keep it out of gold training until the intended
  meaning is decided.
- **Out of domain:** no separate OOD set or OOD label is present.

### `conversation`

- **Belongs:** non-operation/general conversational requests in the current
  label convention. Phase 8 says this intent is routing only.
- **Does not belong:** explicit requests to search, inspect the screen, type
  supplied text, or perform another named operation.
- **Positive:** “Tell me a joke” (`dataset.py`). This is a training label, not
  evidence that a response generator currently exists.
- **Negative:** “Search YouTube for Python tutorials” (`dataset.py`,
  `search_web`).
- **Ambiguous/review:** “Compose a polite reply to this email”
  (`type_text_v2.jsonl`, labeled `conversation`) does not specify whether Mira
  should generate a reply, type provided text, or both. Those behaviors are
  not defined by the current intent interface; adjudicate or keep as a challenge
  case rather than silently choosing a new capability.
- **Out of domain:** no separate OOD set or OOD label is present.

### Cross-Class Boundary Rules

- **Open vs search:** “Launch Chrome” is `open_application`; “Search YouTube
  for Python tutorials” is `search_web`. An application/service name alone
  does not determine the label.
- **Read screen vs conversation:** “What does this window say?” is
  `read_screen`; an unanchored topical question is not screen reading. The
  existing screen class is text-only classification; the model receives no
  screenshot.
- **Type text vs conversation:** entering text is `type_text`; asking Mira to
  compose text is labeled `conversation` in the current sample but generation
  behavior is undefined. Examples that could mean either must be reviewed.
- **File operation vs system control:** local files/folders are
  `file_operation`; power, lock, volume, and brightness examples are
  `system_control`.
- **System control vs conversation:** an imperative to change device state is
  `system_control`; social/general utterances are `conversation`.
- **Ambiguous app references:** entity resolution is downstream. A request can
  clearly mean “open an application” while the application identity is unknown
  or ambiguous. Do not change its intent label based on `apps.json` membership.
- **Short/long/casual/imperfect English:** include natural short commands,
  longer polite requests, casual speech, varied word order, and realistic
  imperfect/ASR-like transcripts. Current examples are mostly short, clean
  English; there is no evidence of accent-specific or ASR-transcript coverage.
- **Multi-intent:** `train.py` requires exactly one label. Existing rows include
  compound requests (examples noted above), but the repository defines no
  decomposition/primary-intent rule. Mark and adjudicate these; do not use an
  arbitrary single label as if the policy were settled.
- **OOD:** no OOD examples, split, metric, or supervised `Unknown` class is
  present. A future OOD challenge set and its expected outcome require a
  separate decision; this datasheet does not invent sample utterances.

## Dataset Inventory and Current State

All counts below are observed from the current JSONL files/report; they are
not recommended dataset sizes.

| File | Rows | Observed labels / note |
|---|---:|---|
| `models/training/corpus.jsonl` | 670 | 310 `open_application`, 310 `search_web`, 10 for each other intent |
| `models/training/train.jsonl` | 536 | 248 each `open_application` and `search_web`; 8 for each other intent |
| `models/training/validation.jsonl` | 67 | 31 each `open_application` and `search_web`; 1 for each other intent |
| `models/training/test.jsonl` | 67 | 31 each `open_application` and `search_web`; 1 for each other intent |
| `datasets/open_application.jsonl` | 1,165 | All `open_application` |
| `datasets/search_web.jsonl` | 1,016 | All `search_web` |
| `datasets/conversation_v2.jsonl` | 72 | All `conversation` |
| `datasets/file_operation_v2.jsonl` | 72 | All `file_operation` |
| `datasets/read_screen_v2.jsonl` | 72 | All `read_screen` |
| `datasets/system_control_v2.jsonl` | 72 | All `system_control` |
| `datasets/type_text_v2.jsonl` | 72 | 69 `type_text`, 3 `conversation` |
| `datasets/hard_negatives_v2.jsonl` | 30 | All labeled `open_application`; despite the filename these are not negative-class rows |

`build_splits.py` reads only `dataset.py` (70 curated rows) and the original
`open_application.jsonl` / `search_web.jsonl`. It does **not** read any `_v2`
file or `hard_negatives_v2.jsonl`. The checked-in `split_report.json` reports
12 exact normalized duplicates dropped, 1,569 generated rows dropped by the
300-per-intent cap, 670 corpus rows, and zero overlaps under its
lowercase/whitespace/punctuation-stripped text check. Its emitted records have
only `text` and `label`; source/group information is not retained. The split is
row-random within each class, not source/template-grouped. The five smaller
classes consequently have only one example apiece in validation and test.

The saved `reports/eval_baseline_test.json` reports 67 test examples, accuracy
0.7612, macro-F1 0.5550, balanced accuracy 0.7880. Its support is 31 for each
of `open_application` and `search_web`, and 1 for each of the other five
intents. This is a recorded result on a very small/imbalanced per-class test
set, not a robust quality estimate.

## Dataset Requirements and Quality Rules

The following requirements are grounded in the Phase 7 section of
`ROADMAP.md`; numerical ratios or minimum counts not specified there remain
undecided.

- Include all seven canonical intents and balance train/validation/test per
  intent. Report counts after deduplication. Do not treat the current 248-vs-8
  train split or 31-vs-1 test split as balanced.
- Cover realistic commands, different word orders, short and long requests,
  casual phrasing, imperfect English, ambiguity, hard negatives, and varied
  natural-language forms. Current corpus coverage must be audited rather than
  inferred from filenames or row counts.
- Preserve actual user-request text; a JSONL training record currently has
  `text` (non-empty string) and `label` (one exact canonical string). Extra
  metadata may be used for preparation, but `train.py` ignores it and emits
  only the two fields when its data is built.
- Reject malformed JSON, missing fields, wrong field types, empty/whitespace
  `text`, and labels outside the seven-name catalog. Current `train.py`
  blindly parses rows and indexes `OPTIONS`; it does not perform these checks.
- Exact duplicates: normalize case and whitespace and compare punctuation-
  insensitive text. Conflicting labels for the same normalized text are an
  annotation error to resolve, not two valid samples. `build_splits.py` drops
  exact lowercase/whitespace text+same-label duplicates, then checks
  punctuation-stripped split overlap; this is not a full Unicode-aware audit.
- Near duplicates: identify paraphrase/template families and keep all related
  variants in one split or remove redundant variants. The repository builder
  does not perform semantic or template-family detection.
- Split isolation: deduplicate and assign source/template/semantic groups
  before splitting. Ensure no group, paraphrase family, or derived example
  crosses train/validation/test. Keep the test split fixed and unused for
  model/threshold selection. Current builder does row-level stratified random
  splits and cannot enforce source/template isolation because it drops source
  metadata.
- Label correctness: apply the definitions and boundary decisions above;
  manually adjudicate conflicting, compound, unclear, and high-impact examples.
  Do not infer label correctness from a filename, generator, or majority class.
- Balance should reflect natural coverage within each class as well as class
  counts; do not achieve it through copies or slot substitutions alone.

No minimum number of examples per class, fixed balance tolerance, or split
percentage is specified by the model interface or current Phase 7 criteria.
`build_splits.py` currently uses 80/10/10 by row count, seed 42, and a generated
cap of 300 for each of two classes; these are script settings, not validated
quality requirements.

## Training, Evaluation, and Unknown Behavior

- `train.py` fine-tunes from `../tamev-base` with PyTorch/Transformers and
  cross-entropy over the index of the fixed seven-option list. Its configured
  batch size is 4, epochs 10, learning rate `2e-5`, context limit 128, option
  limit 64, and CPU device. These are current script settings, not datasheet
  recommendations.
- `train.py` selects checkpoints by validation accuracy and reports train/val
  loss and validation accuracy, then test loss/accuracy. It does not call
  `metrics.py` for per-intent results.
- `eval_model.py` and `metrics.py` support accuracy, per-intent precision,
  recall, F1, macro-F1, balanced accuracy, and a confusion matrix. Use them on
  both validation and held-out test data; report support per class. The current
  test support makes per-intent estimates for five classes uninformative.
- `TamevClassifier` has a configurable confidence threshold, default 0.5, and
  returns `Intent::Unknown` when the maximum probability is below it or
  inference is unavailable/fails. `Unknown` is not in the seven-option output;
  the training/evaluation scripts do not measure OOD rejection or calibrate the
  threshold. Softmax confidence alone does not establish OOD recognition.
- The Phase 7 roadmap asks for an exported model that measurably beats the
  previous one and for Phase 6 parity after export. The repo defines no
  numerical acceptable accuracy/F1 threshold or statistical test for
  “measurably.” Report the same locked test set and baseline/candidate metrics;
  choose any stronger threshold before evaluating, not after seeing test
  results.
- The saved baseline report is the only current numeric quality result found;
  it is described above. Do not represent the near-uniform base ONNX parity
  vectors as task-quality evaluation.
- Before runtime use, verify the fine-tuned exported artifact against Python
  and the canonical C++ context, tokenizer, option order, and lengths. The
  current parity artifact is the base model, not the fine-tuned checkpoint.

## Phase 7 Acceptance Criteria

The repository-supported completion gates are:

1. The dataset integrity checks pass for schema, canonical labels, duplicates,
  split disjointness, and class-balance reporting.
2. Per-intent precision/recall and the confusion matrix are reported, along
  with aggregate accuracy/F1 metrics, on validation and the held-out test set.
3. The candidate fine-tuned model measurably beats the previous model on the
  same fixed test set, as required by `ROADMAP.md` Phase 7. The current saved
  baseline is 0.7612 accuracy and 0.5550 macro-F1 on its 67-row test split;
  because that test split has only one example in five classes, it is not an
  adequate final acceptance set without improved per-class support.
4. The ONNX export is checked against PyTorch on a fixed sample, and the Phase
  6 Python/C++ parity harness passes on the fine-tuned artifact, not merely on
  the downloaded base model.
5. The option order, context format, tokenizer, and tensor lengths used in
  training, evaluation, export, and C++ inference are consistent.

No numerical threshold for “measurably beats,” confidence interval/significance
test, minimum per-class test support, absolute accuracy/F1 floor, or OOD false
acceptance limit is specified in the repository. Those must be agreed and
recorded before the final evaluation; this datasheet deliberately does not
invent them. The runtime classifier's default 0.5 confidence threshold is an
implementation default, not a validated acceptance threshold.

## Current Limitations and Decisions Still Open

- Current `train.py` consumes split files, not the raw `datasets/` directory.
  The split builder currently omits the `_v2` files, including the five
  additional per-intent corpora and `hard_negatives_v2.jsonl`.
- `hard_negatives_v2.jsonl` contains only `open_application` labels, so it is
  not currently a cross-class hard-negative set. `type_text_v2.jsonl` contains
  three `conversation` labels that need review.
- Split reporting's zero punctuation-stripped overlap does not establish
  semantic/template/source leakage absence.
- `models/tester.py` / `models/runners/tester.py` disagree with the canonical
  seven-option training/runtime contract. `models/tamev-base/config.json`
  reports option length 48 while the training/runtime contract uses 64.
- Current saved test metrics have inadequate per-class support for five
  classes. No acceptable absolute metric thresholds or OOD policy are defined.
- `README.md` and some older roadmap inventory statements describe earlier
  repository states (for example that the resolver and C++ classifier are
  absent). This datasheet follows the present source files for model/data
  details and the current Phase 7 roadmap for project acceptance direction.
- No OOD/Unknown training examples, multi-intent resolution policy, language
  coverage target, per-class minimum size, numerical balance tolerance, or
  statistically defined improvement threshold is supplied by the repository.
  These are decisions to make before data generation or acceptance testing.