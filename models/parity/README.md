# models/parity — frozen parity vectors for Phase 6

`frozen_vectors.json` pins the exact expected output of the fixed ONNX
artifact `models/model_int8.onnx` for a frozen utterance list, using the
canonical Phase 6 inputs:

- context = `"Which action should Mira perform?\nUser request: " + utterance`
  encoded with `models/tamev-base/tokenizer.json`, padded to ctx_len = 128
- options = the 7-entry catalog in `tamev_option_strings()` training order,
  each padded to opt_len = 64

Regenerate with (`v.sh` activates the project venv, ORT 1.22 per that env):

```
source ~/v.sh        # or: source /home/arc/v.sh  (activates ~/ml_env)
python3 models/parity/regenerate.py
```

`regenerate.py` prints the generator onnxruntime version into stdout and
stores it under the `generator` key of the JSON, so any ORT-version skew is
visible. When the venv ORT differs from the system ORT the C++ binary links
(1.29), int8 kernels may differ slightly — see the tolerance note in
`tests/test_tamev_parity.cpp`.

See `tests/test_tamev_parity.cpp` (argmax equality is the hard gate; per-entry
probabilities must match within 1e-3 — measured venv-ORT 1.30 vs system-ORT
1.29 int8 kernel skew is 4.0e-04, so 1e-3 tolerates the known skew while still
catching silent drift).

NOTE: this artifact is the downloaded *base* TAMEV checkpoint, not the
fine-tuned `training/mira-tamev` model — vectors are near-uniform by design.
Parity here proves the C++ path reproduces Python exactly; wiring the
fine-tuned model is Phase 7 work.
