# The MLX backend: what was taken from laya-mlx, and what was not

September 2026. Source: a full read of `/Users/yogthos/src/laya-mlx` (the
independent MLX port of the same convaiinnovations/laya checkpoints lev
runs), its BENCHMARKS.md, its `experiments/` (the 10x research: math
budgets, weight spectra, compile / quantization / head-pruning / custom
Metal kernel trials) and lev's own seams. The measurements this note
promised are in bench/README.md; this is the record of the decisions.

## What laya-mlx is

A from-scratch MLX inference runtime for the exact checkpoints lev serves:
ModernBERT-large / mmBERT-base encoder, 2-layer transformer decision head,
scorer + act head, RLCD calibration read from the checkpoint's JSON,
tokenized with HF's Rust `tokenizers`, answering in the shape lev's
`system-one` already produces. lev already had every piece of that
runtime, implemented independently and held to a tighter parity bar (one
unit in the fourth decimal against the torch oracle, where laya-mlx's
FP16 drifts 5e-3), and lev's C attention already skips out-of-window key
tiles on sliding layers, which laya-mlx's math note only proposes. So
nothing of the runtime was ported as code. Four things transferred.

## 1. The MLX runtime itself, as a native backend (landed: `lev.mlx`)

The first draft of this note recommended a Python sidecar, believing MLX
had no stable C face. It does: [mlx-c](https://github.com/ml-explore/mlx-c)
is Apple's official C binding (what mlx-swift sits on), released in
lockstep with MLX; its v0.6.0 tag fetches mlx v0.31.1 and exposes
`mlx_fast_rope`, `mlx_fast_scaled_dot_product_attention` (boolean array
mask), `mlx_fast_layer_norm`, `mlx_addmm`, `mlx_erf`, `mlx_take`,
`mlx_array_new_data`, `mlx_eval`. That is the whole forward, so the
backend took exactly the shape `lev_llm.c` already had for llama.cpp:

- `jolt mlx` clones mlx-c at the pin into `native/mlx-c`, cmake-builds it
  static (its FetchContent pulls MLX at the matching tag), and
  `native/build_mlx.sh` wraps `native/lev_mlx.c` with both archives into
  `liblev_mlx.{dylib,a}`; `:optional true` in `:jolt/native`, darwin
  only. A tree without it runs exactly as before.
- `native/lev_mlx.c` holds the weights on the device (read from lev's own
  `data/*.f32` files, one `lev_mlx_load_tensor` per manifest entry, so
  the 1.7 GB never enters the jolt heap) and builds the forward as one
  lazy graph per batch, evaluated once: embeddings, N layers (RoPE,
  fast SDPA with the padding / window mask, exact-erf gated GELU), final
  norm, type embedding, the two head layers, the scorer at the markers,
  the act features and head. What crosses the FFI is what
  `lev.model/forward-batch` answers: the marker logits and two act
  logits per row, f32.
- The seam is `forward-batch`, not `system-one*`: an encoder agent now
  carries `:forward` (the C kernels' `lev.model/forward-batch` by default)
  and `lev.mlx/load-agent` builds the same agent shell
  (`lev.agent/agent-shell`: config, tokenizer, prefix cache, limits) with
  the device model as `:w` and `lev.mlx/forward-batch` as `:forward`. So
  validation, sequence building, calibration, constraints, debias,
  truncation reporting and `lev.calibrate` are shared, untouched.
- Config: `:backend cpu|mlx` and `:dtype f32|f16`, top-level or per
  checkpoint, env `LEV_BACKEND` / `LEV_DTYPE`, CLI `--backend` /
  `--dtype`, through `lev.config/limits` like the sequence limits; the
  router's `load-prepared` dispatches on it and answers
  `:model-unavailable` when the native is not built.
- Agents that own native state now carry `:close`, and the router calls
  it on eviction and unload — which also fixed a pre-existing leak: an
  evicted CPU agent's 1.7 GB of malloc'd weights and an unloaded
  thinker's llama context were never freed.

### The numerical-parity decision

laya-mlx's FP16 passes *task* parity (argmax agreement) but not *value*
parity (5.4e-3 drift); lev's bar is value parity. So the backend
defaults to f32 on the device, where it reproduces the C kernels on the
golden batch to 1e-4 and the README answers to the oracle
(`lev.mlx-test`), and authored144 / the trio to the case and the ECE
digit. f16 is opt-in: the same labels (one authored144 case moves, in
its favour), probabilities within 1e-2, half the memory, ~20% faster
again. bench/README.md has the table: **4.0x** on a short four-question
call at f32, 5.1x at f16, measured interleaved.

### The metallib

MLX's Metal kernels are a 100 MB `mlx.metallib` it loads from next to the
binary holding MLX, else from the build tree's absolute path. The build
puts a copy in `native/` (next to the dylib, for `jolt run` / `test`);
a `jolt build` binary needs it beside `lev-server`. A missing library is
reported as such by `lev.mlx` with that hint (the first MLX error is
kept, since every op after it fails with an unhelpful "empty array").

## 2. The state tokenized once, the prefixes cached (landed: `lev.sequence`)

laya-mlx's `PrefixCache`: cache the tokenized question prefix `[CLS]
<type> instructions [SEP] [MASK] opt0 ... [SEP]` keyed by (head_max_len,
type, instructions, options), and tokenize the state once per call. lev
was tokenizing the state once *per question*, at ~33 µs a token: 48 ms a
question on a 1.6k-token state, four times over in a four-question call.
Now `build-prefix` / `encode-state` / `assemble` split `build-sequence`
in two and the agent keeps a 128-entry LRU of prefixes. Byte-identical
ids (the sequence golden proves it); the long-state four-question call
drops 10%.

## 3. The paired benchmark (landed: `bench/paired.clj`)

`experiments/engineering/paired.py` + `analyze.py`: every round takes one
input and runs every candidate on it in an order rotated per round;
report the median of per-round `baseline / candidate` ratios with a
percentile-bootstrap 95% interval. Their sequential pilot said 1.24x for
compile, the paired run 1.03x; lev's own before/after of the same code
moved 288 → 304 ms. The noise floor here (`cpu,cpu`) is 1.00x ±2% over
20 rounds. It is what decides whether the last item lands.

## 4. Exact last-head-layer pruning (measured before landing)

`run_variants.py`'s `selected_head`: only the CLS and marker rows of the
last head layer are read downstream, so its Q, out-projection and FFN
need only those 1+k rows (K/V still all rows). Exact dependency pruning;
2.7–4.6% of modeled FLOPs; laya-mlx measured 1.03–1.08x paired. Lands
only if `bench/paired.clj` puts its interval above 1 on lev.

## Not ported, with the evidence

- `mx.compile`: 1.03x interleaved on their side; nothing to fuse across
  lev's imperative C-kernel forward, and the MLX graph here is built in C
  per batch, where laziness already batches the dispatches.
- A custom exact-erf Metal GELU/gate kernel: bit-exact, no consistent win
  over MLX's compiled expression.
- 8-bit / 4-bit backbone weights: no speedup at these shapes (larger
  pilots slower), 62/63 and 50/63 fixture agreement with probability
  shifts up to 0.33.
- Low-rank weights: the rank a 10x matrix-work cut allows keeps 25–33% of
  the Frobenius energy (84% relative error) on the four sampled matrices.
- Exact whole-input deduplication: lev's questions within a call always
  have distinct prefixes (debias rotations too), so there is no hit.
- First-layer QKV precompute (0.85% of work), `pad_to_multiple` (only
  helps a shape-specializing compiler), length-sorted chunking (rows in a
  call differ only by prefix length, ≤192 tokens).
