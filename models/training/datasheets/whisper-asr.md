# Whisper ASR Datasheet

This datasheet describes only the Whisper inference integration present in
MiraV2 and separates it from any possible future fine-tuning. There is no
Whisper training or export pipeline in this repository.

## Model Identity and Purpose

- **Model:** OpenAI Whisper English base model, loaded through the vendored
  `whisper.cpp` API.
- **Artifact:** `models/ggml-base.en.bin` is the model path referenced by the
  current `main.cpp`. The wrapper accepts a model path in its constructor; the
  path used by `main.cpp` is relative to the process working directory.
- **Purpose/task:** transcribe captured speech audio into text. It does not
  choose an intent, extract entities, generate a response, or perform an
  action.
- **Training status:** no Whisper fine-tuning script, paired audio/transcript
  corpus, Whisper dataset manifest, or Whisper export process was found.

## Current Input and Output

- `Whisper::transcribe` accepts `const std::vector<float>& audio` and returns a
  `std::string` transcript. The wrapper does not accept a sample-rate/channel
  argument or validate audio metadata.
- The microphone implementation/project contract is 16 kHz mono float PCM for
  Whisper. The Whisper wrapper itself simply passes the sample vector to
  `whisper_full`.
- `speech/whisper.cpp` sets `params.language = "en"`, uses greedy sampling,
  four threads, and disables progress output. It reads the Whisper segment
  strings and concatenates them directly, without an inserted separator or
  other transcript normalization.
- Model initialization failure and transcription failure throw
  `std::runtime_error`. No confidence, timestamps, language detection result,
  or structured ASR status is returned by Mira's wrapper.

## Downstream Contract

In the intended architecture, the transcript text is passed to TAMEV for intent
selection. TAMEV decides the intent; the intent-aware Tokenizer runs only after
that decision and extracts the target/query/text. Whisper must not emit or be
trained against a TAMEV label as part of its transcript output.

The current `main.cpp` connects Whisper to TAMEV through `mira::Pipeline`
(Phase 8): the transcript is classified, tokenised and dispatched, with the full
decision chain printed per run. The expected TAMEV
input is the transcript string, wrapped by the canonical TAMEV context
builder as `Which action should Mira perform?\nUser request: <transcript>`.
Whisper itself does not currently clean, lowercase, trim, punctuate, or correct
the returned text after segment concatenation.

## Transcript Expectations and Error Analysis

The repository does not define an annotation/normalization policy for training
transcripts, nor does it contain ASR evaluation results. Therefore it does not
establish whether punctuation, casing, fillers, repetitions, or disfluencies
should be preserved or normalized. Do not silently “correct” transcript text
before passing it to TAMEV without an explicit shared policy and parity tests.

Word substitutions, omissions, insertions, segmentation/spacing differences,
and errors on names or command-critical words are reasonable ASR error
categories to measure because they can change the downstream intent or target.
They are **test categories**, not documented observed Whisper failures in this
repository. No word-error analysis or confidence characterization is present.

## Dataset Requirements if Fine-Tuning Is Later Approved

There is no current Whisper dataset specification or training pipeline, so the
following data format, split, and size are **not defined by repository code**.
At minimum, any later Whisper adaptation would need paired audio and verified
transcript text; text-only JSONL used by TAMEV cannot supervise speech
recognition. Preserve stable audio references and record enough metadata to
reproduce decoding/resampling and group recordings by speaker/session/source.
Transcript conventions and the handling of silence, partial/unintelligible
speech, multilingual audio, and synthetic speech must be decided before
collection or annotation. Keep related recordings, crops, and augmentations in
one evaluation split to prevent leakage. The repository specifies no minimum
hours, speaker count, train/validation/test ratio, or collection policy; none is
asserted here.

Do not add TAMEV intent labels as ASR targets. If utterances are also annotated
for downstream evaluation, store that intent annotation separately and keep
the ASR transcript target verbatim under the selected annotation policy.

## Evaluation

No ASR metric is currently implemented or reported in MiraV2. Word Error Rate
(WER) is appropriate for transcript word accuracy; Character Error Rate (CER)
can diagnose spelling/tokenization errors. For the Mira use case, separately
inspect exact or error rates on action words and named entities because an
otherwise low WER can conceal an intent-changing substitution. Report the text
normalization used for scoring and stratify results by conditions represented
in the evaluation data. Evaluate downstream TAMEV on transcripts from held-out
audio as well as on clean text if an ASR adaptation is introduced.

No numeric WER/CER threshold or current baseline score is specified in the
repository. A measurable acceptance threshold and test composition must be
selected before a future ASR training/evaluation run; this datasheet does not
invent one.

## Current Limitations vs Future Plans

**Current implementation:** pretrained `base.en` inference only; English is
explicitly selected; greedy decoding; four threads; float-vector input; string
output created by direct segment concatenation; no transcript post-processing;
no training/evaluation dataset or metrics found.

**Not current behavior:** fine-tuning Whisper, dataset generation, language
adaptation, confidence-based rejection, VAD/endpointing inside the wrapper,
transcript normalization, entity correction, or TAMEV invocation from
`main.cpp`. These require separate future design and implementation.