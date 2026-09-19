# laya-jolt

Port of the Laya decision-model inference engine (Python/torch in
`../laya`: `rl_agent_api.py` + `rl_common.py`) to jolt. The model never
generates text: state + typed questions in, calibrated typed answers out.

## Decisions

- **float32 everywhere.** Weights are F16 in `model.safetensors`; F16->F32 is
  an exact upcast, done once at prepare time into raw f32 blobs under
  `data/`. No mixed precision in the port; the Python oracle also runs CPU
  float32 so the tolerance budget is small.
- **Golden-trace oracle.** `python/dump_traces.py` (in `.venv`, torch CPU f32)
  dumps input_ids, embeddings, every encoder layer output, head intermediates,
  logits and act outputs for a fixed case set under `golden/`. Every jolt
  stage is tested against these to tolerances recorded in the tests
  (~1e-4 relative on logits, tighter layer-by-layer).
- **NFC normalization via ICU.** The tokenizer's normalizer is `{"type":"NFC"}`.
  We FFI to `unorm2_getNFCInstance` / `unorm2_normalize` (mac:
  `libicucore.dylib`; linux: `libicuuc.so*`). UTF-8 <-> UTF-16 conversion is
  done jolt-side; ICU gets UTF-16 buffers.
- **BLAS via FFI.** `cblas_sgemm` from Accelerate (mac) or OpenBLAS (linux)
  for every matmul.
- **C kernels where BLAS doesn't reach.** `native/laya_kernels.c` (compiled by
  the `kernels` task, bound via `jolt.ffi`) holds the elementwise/reduction
  ops: rmsnorm, layernorm-with-bias, softmax (masked), sliding-window mask
  build, gelu (erf), relu, silu, swiglu, rope-apply, marker gather. Nothing
  per-element crosses the FFI boundary (the naval-battle rule).
- **The head FF uses ReLU.** `nn.TransformerEncoderLayer(..., norm_first=True)`
  defaults `activation="relu"`. There is a standing task to verify this against
  the torch source in the venv AND with a golden trace that fails if GELU is
  used. Encoder SwiGLU activation is verified the same way from
  `modeling_modernbert.py`.
- **Linux is a first-class target.** All native deps have darwin+linux entries
  in `deps.edn`; the build tasks branch on OS. (CoreML is macOS-only and
  therefore NOT used: see the deferred task on the board. If it is ever tried
  for mac speed it must reproduce answers within the same tolerances.)
- **One row per forward, unpadded.** Python batches all questions padded to
  the longest sequence (`collate_items`); with correct key-padding masks the
  real positions come out the same either way, so the port runs each
  question at its own length and skips padding entirely. Padding-mask
  semantics are still verified: the golden batch has a padded row, and the
  encoder/head parity tests check both rows.
- **Python string semantics are part of the contract.** The state is
  `json.dumps`'d before tokenization, so `laya.sequence/json-str` reproduces
  Python's float repr, escapes and `ensure_ascii`; `laya.email` reproduces
  `str.isspace` / `\w` / `strip` where irregex's ASCII classes would not.
- **Per-call arenas.** Every intermediate tensor of a forward pass is owned
  by an `ffi` arena bound in `model/forward-row`, released on return.
  Weights and cached rope tables are allocated outside it.

- **HTTP API mirrors TypeSafe.** `laya.server` is a ring handler (library
  use) and a `-main` (server / `jolt build` binary) for `POST /v1/systemone`
  with Bearer auth and FastAPI-style 422 details. Requests are read by
  `laya.json`, which keeps object key order (option/question/state-field
  order is model input; `data.json` cannot keep it).
- **The AOT binary self-tests.** jolt 0.8.9 release builds miscompile
  `(reduce (fn [acc x] (if (or (nil? acc) ...) ...)) nil xs)` (the nil? folds
  to true). The code avoids the pattern and `jolt binary` runs
  `./laya-server --self-test` against `golden/` after every build.

## Layout

- `src/laya/` — tensors (matmul/kernels), tokenizer, model (encoder + head),
  sequence building, agent API, email helpers, checkpoint conversion.
- `test/laya/` — clojure.test parity suites + `laya.test-runner`.
- `python/` — the oracles: `dump_traces.py` (the `.venv` python with torch)
  and `prepare.py` (numpy), the reference converter whose output is pinned
  in `golden/prepare.edn`.
- `data/` — generated f32 blobs, `manifest.edn`, `tokenizer.edn`,
  `config.edn`. Gitignored.
- `golden/` — dumped traces. Tracked (small, they are the contract).

## Commands

- `jolt kernels` — compile `native/laya_kernels.c`
- `jolt prepare` — convert the `../laya` checkpoint into `data/` (pure jolt)
- `jolt traces` — re-dump golden traces and prepare checksums with the venv python
- `jolt -M:test` — run the parity suites
- `jolt -M:run` — quickstart demo from the README
- `jolt -M:serve` — the HTTP API; `jolt binary` — standalone `./laya-server`
