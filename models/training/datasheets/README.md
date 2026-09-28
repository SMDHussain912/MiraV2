# Model Dataset Datasheets

These specifications describe data needed to train or fine-tune models already
present in, or directly used by, MiraV2. They are requirements for future data
work only. They do not contain generated corpora, authorize training, or change
runtime behavior.

## Model Inventory

| Model / artifact | Task | Dataset status | Datasheet |
|---|---|---|---|
| Mira-TAMEV fine-tune (`models/training/mira-tamev/`) | Select one of seven intents from a text request | A 70-row hand-authored corpus is split into 56/7/7 rows; larger 1,165-row `open_application` and 1,016-row `search_web` corpora exist but are not consumed by `train.py`. | [TAMEV intent classifier](tamev-intent-classifier.md) |
| TAMEV released/base checkpoint (`models/model_int8.onnx`; related base configuration/tokenizer in `models/tamev-base/`) | Same option-selection interface; upstream artifact, not Mira's fine-tuned checkpoint | Frozen vectors prove C++/Python inference parity for this downloaded base artifact, not useful task accuracy. It uses the same dataset contract if adapted for Mira. | [TAMEV intent classifier](tamev-intent-classifier.md) |
| Whisper `base.en` (`models/ggml-base.en.bin`) | English speech-to-text | Pretrained runtime artifact. No Whisper training/fine-tuning pipeline or paired speech corpus is present. A speech dataset requires audio plus text transcripts, not text alone. | [Whisper ASR](whisper-asr.md) |

The TinyBERT encoder named in `models/tamev-base/config.json` is an upstream
pretrained backbone. This repository does not train it from scratch or provide
its original pretraining corpus; it is not a separate Mira task dataset.

The roadmap's future OCR/vision work has no selected model or training
interface. The roadmap explicitly defers choosing an OCR engine, and Tesseract
is not installed in the documented environment. Do not generate an OCR dataset
or infer a model datasheet until that model and its interface are selected.

## Repository Findings

- TAMEV training examples currently use exactly `text` and `label`, with seven
  lowercase snake_case labels. The label becomes the index in a fixed candidate
  list; candidate ordering is part of the model contract.
- The current small split contains 8 examples per class in train and 1 per
  class in validation and test. This is a smoke-sized corpus, not an adequate
  estimate of generalization.
- The two larger JSONL files cover only `open_application` and `search_web`.
  They cannot alone train the seven-way classifier. A simple case/punctuation
  normalization audit found 2 duplicate rows in the former and 7 in the latter;
  semantic/template leakage has not been audited.
- `models/tester.py` uses a different question string and only four options.
  The task data contract should follow the training and Phase 6 parity format
  documented in the TAMEV datasheet; the existing inference script mismatch is
  recorded here, not fixed by this documentation change.
- `speech/whisper.cpp` expects audio samples and currently fixes language to
  English. It has no text-only training interface.

Minimum sizes and thresholds in the datasheets are proposed recommendations,
not claims about current model performance or promises that a particular size
will guarantee quality.