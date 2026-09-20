# Adopting an optional MLX backend for lev

September 2026. Source: a full read of `/Users/yogthos/src/laya-mlx` (the
independent MLX port of the same convaiinnovations/laya checkpoints lev
runs), its BENCHMARKS.md, and lev's own seams (lev.agent, lev.model,
lev.router, lev.sequence, native/lev_llm.c + build_llm.sh, deps.edn).

## What laya-mlx is

A from-scratch MLX inference runtime for the exact checkpoints lev serves:
ModernBERT-large / mmBERT-base encoder, 2-layer transformer decision head,
scorer + act head, RLCD calibration read from the checkpoint's JSON. It
loads the *upstream* safetensors directly (parameter names sanitized at
load), tokenizes with `tokenizers`, and answers in the same shape lev's
`system-one` already produces (choice/score/noul, probabilities rounded to
4 decimals, entropy confidence, act_probability). No PyTorch anywhere.

## What the numbers say

Measured on an M3 Max (laya-mlx's BENCHMARKS.md; lev runs on the same
class of machine). End-to-end latency per request, P50:

- **1 question, short input, FP16**: laya 13.4 ms, multilingual 7.4 ms,
  typed-decisions 13.7 ms.
- **10 questions FP16**: 71 ms / 27 ms / 76 ms.
- **Full-context 512/1024-token single question FP16**: 45–99 ms.
- FP32 MLX is ~1.2–1.6x slower than FP16; PyTorch MPS FP32 slower still.
- Accuracy: 256/256 AG News prediction agreement with upstream FP32 at
  FP16, max calibrated probability error 0.0054 (lev's parity bar is one
  unit in the fourth decimal, ~0.0001, so **FP16 fails lev's golden
  tests as-is**; FP32 MLX agrees to 5e-6).
- Memory: FP16 weights are 803.6 MiB for the 421M checkpoints (vs ~1.6
  GiB f32), peak ~944 MiB for one short question.

Compare lev today: ~125 ms a case on the M-series CPU (bench/README),
f32, one binary, no Python. So the prize is real but bounded: roughly
**3–9x on short inputs, ~2x on full context, half the memory** — in
exchange for the costs below.

## Three integration paths, ranked

### 1. (Recommended) a sidecar process, like laya-mlx's own snake demo

lev-server keeps its C kernels; an optional MLX sidecar (the laya-mlx
package as-is, plus a ~100-line stdin/stdout or HTTP shim exposing
`prepare`/`forward` per batch) is spawned when config asks for it.
The router already has the seam: `:loader` in make-router. A
`load-prepared` variant that returns a `{:kind :mlx ...}` agent map, and
a `defmethod system-one* :mlx` that ships the already-validated,
already-built sequences to the sidecar and gets logits back, leaves
*every* lev-side invariant intact: sequence building, calibration,
constraints, debias, truncation reporting. The logits are the only thing
that crosses the boundary, and they're f32 (sidecar converts FP16
internals to f32 before returning).

- Pros: zero new native code in lev, no llama.cpp-scale build matrix, the
  421M params never enter the jolt heap, MLX stays optional (a tree
  without uv/laya-mlx runs exactly as today), and the parity story is
  clean — golden tests compare logits sidecar-vs-C-kernels to 1e-4.
- Cons: a Python process in the deployment story; one pipe round-trip
  per batch (~0.1–0.5 ms, noise vs 13 ms); on mac only (Metal; the
  linux story would be CUDA-only MLX or nothing).

### 2. A native lev_mlx.c shim over libmlx.dylib

Mirror lev_llm.c: a flat C face over `mx.array` / `mx.fast.rope` /
`mx.fast.scaled_dot_product_attention`, declared in deps.edn :jolt/native,
built by a `jolt mlx` task cloning the mlx-cxx repo at a pin. This is the
"one binary" purist path but it is a *large* lift: mlx's C++ API is
template-heavy and not C-stable; every op would need hand-written
wrappers; weights would flow through jolt buffers. Cost is weeks, not
days. Only worth it if path 1 proves out and the sidecar process offends.

### 3. Port MLX's *ideas* only, no MLX dependency

Two directly portable wins, both applicable to lev's CPU backend today:

- **PrefixCache (laya_mlx/prepared.py)**: cache the tokenized question
  prefix `[CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP]` keyed by
  (tokenizer, head_max_len, qtype, instructions, options). lev's
  sequences are built identically (lev.sequence/build-sequence), and its
  workloads are exactly the hit pattern: same questions, fresh state —
  the typed-decisions workflows, authored144's repeated templates,
  snake's per-tick questions. laya-mlx bounds it at 128 entries LRU.
  Saves the whole tokenize+build cost of every repeat; at 125 ms a case
  even 10% is measurable. **Cheapest real win in this document.**
- **pad_to_multiple bucketing**: pad rows to a multiple of 16 tokens so
  MLX's compiler can reuse kernels across calls. On lev's C kernels this
  is neutral at best (its sgemm is shape-generic); skip unless measured.

Not portable: MLX's `mx.compile` graph fusion — lev's forward is already
imperative Clojure over C kernels; there is nothing to fuse across.

## What does NOT transfer from laya-mlx's implementation

lev already implements, independently and to tighter parity, everything
laya-mlx had to build: sequence layout, option rendering, noul [false
true] ordering, confidence_from_probs, temp_bucket calibration buckets,
collation (lev's forward-batch pads per-chunk already), act head softmax,
answer shapes. Porting code would be a regression risk with no gain.
The only genuinely new artifacts worth taking are the two optimizations
above and the snake demo (being ported separately as examples/snake).

## Numerical-parity decision lev has to make

laya-mlx's FP16 passes *task* parity (256/256 argmax agreement) but not
*value* parity (5.4e-3 probability drift). lev's convention ("compared
to one unit in the fourth decimal") is a value-parity bar. So an MLX
backend must default to FP32 on the weights (5e-6 agreement, well under
the bar) and treat FP16 as an opt-in speed mode, documented as
approximate, off the golden path — the same posture lev already takes
for the thinker's greedy decoding vs the encoders' exact answers.

## Recommended sequence

1. Port PrefixCache into lev.sequence (pure Clojure, LRU, keyed as
   above) + a bench/authored144 run before/after. Independent of MLX.
2. examples/snake (in flight) as the end-to-end consumer of the API.
3. Path 1 sidecar behind a config flag (`:backends {:mlx ...}`), with a
   golden logit-parity test (sidecar vs C kernels, tol 1e-4) gating it.
4. Only then decide FP16: bench authored144 + trio accuracy/ECE at FP16
   sidecar vs f32 C kernels; land only if accuracy holds and the README
   documents the drift.

None of steps 3–4 block the snake example; steps 1–2 are useful alone.
