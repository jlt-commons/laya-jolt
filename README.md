# laya-jolt

Pure-Clojure inference for the Laya decision model, running on
[jolt](https://github.com/jolt-lang/jolt) (Chez Scheme, no JVM). Same weights,
same outputs: the stack reproduces the Python engine's `RLAgent.system_one`
answer for the README quickstart byte-for-byte.

The model is a ModernBERT-large encoder (28 layers, RoPE, alternating
full/sliding attention, SwiGLU) plus a 2-layer decision head, a scorer, and an
act head. It does not generate text: it consumes a serialized `state` and a set
of typed questions, and returns calibrated typed answers.

Everything is f32 end to end. F16 checkpoint weights are widened to f32 once,
during `prepare`, so the numerics match the torch CPU oracle exactly.

## Layout

- `native/laya_kernels.c` — the tensor kernels (layernorm, gelu, rope,
  attention, softmax, ...). Everything elementwise stays in C.
- `src/laya/tensors.clj` — f32 tensor views over FFI buffers; `cblas_sgemm`.
- `src/laya/model.clj` — encoder + head forward pass.
- `src/laya/tokenizer.clj` — GPT-2 byte-level BPE with ICU NFC.
- `src/laya/sequence.clj` — `build_sequence` (question -> markers + ids) and
  a `json.dumps`-compatible serializer (the state JSON is tokenized, so its
  bytes matter: Python float repr, escapes, `ensure_ascii`).
- `src/laya/agent.clj` — `system_one` (temperature calibration, confidence).
- `src/laya/email.clj` — `email_utils.py`: `clean-email-body`, `email-state`,
  `email-questions`.
- `src/laya/prepare.clj` — `jolt prepare`: checkpoint -> `data/` (f32 blobs +
  EDN manifest/tokenizer/config), no Python involved.
- `python/dump_traces.py` — dump golden traces from the torch oracle into
  `golden/`. These are the contract the port is tested against.
- `python/prepare.py` — the reference converter; `golden/prepare.edn` pins
  the size and CRC-32 of everything it writes, and the test suite checks that
  `jolt prepare` reproduces them byte for byte.

## Build and run

```
jolt kernels             # compile native/laya_kernels.c
jolt prepare             # ../laya checkpoint -> data/   (LAYA_HOME to point elsewhere)
jolt -M:test             # parity suites vs golden/
jolt -M:run              # README quickstart demo
```

`jolt kernels` shells out to `cc`. `jolt prepare` needs only the checkpoint
(`LAYA_HOME`, default `../laya`) and the kernel library; it runs in a few
seconds. `jolt traces` (regenerating `golden/`) is the one step that needs
the `.venv` python with `torch`, `transformers`, `safetensors` and `numpy`.

`jolt -M:run` prints the quickstart answer JSON. It should be identical to the
`:system-one` value in `golden/readme.edn`.

Answers come back as ordered maps with string keys, in the shape of the
Python dicts. Because option order and question order are part of the model
input, pass `:criteria` and the questions map as ordered maps (`array-map`,
or a literal with at most 8 entries); a hash-map would reorder them.

```clojure
(require '[laya.agent :as ag] '[laya.email :as email])
(def agent (ag/load-agent "data"))
(ag/system-one agent
               (email/email-state "Duplicate billing" raw-body :sender "customer@acme.com")
               (email/email-questions))
```

## Native dependencies

Both platforms are supported; `deps.edn` carries darwin and linux entries and
the build task branches on OS.

- **kernels** — `native/liblaya_kernels.dylib` (mac) / `.so` (linux), built by
  `jolt kernels`.
- **NFC** — `libicucore.dylib` on mac (unguarded symbols in the system dylib),
  `libicuuc.so.<ver>` on linux. The tokenizer calls `unorm2` and the
  `u_charType` / `u_isUWhiteSpace` classifiers. Linux ICU builds append the
  major version to every symbol (`u_charType_76`); the bindings resolve the
  first spelling that exists, for versions 60..90.
- **JSON** — `org.clojure/data.json` from Maven, plus `jolt-lang/time` which
  provides the `java.time` classes data.json needs to load. Only `prepare`
  uses them.
- **BLAS** — `cblas_sgemm` from the Accelerate framework on mac, OpenBLAS on
  linux.

On linux, install the ICU and OpenBLAS runtime packages and adjust the version
suffixes in `deps.edn` if your distro's `libicuuc.so` version is not listed.

## Status

Encoder, head, tokenizer, sequence, agent, email helpers and the checkpoint
conversion all match their golden traces. `system-one` on the quickstart case
is byte-identical to the Python output.

Per-forward temporaries live in an ffi arena that closes with the call, so a
long-running process stays at the size of the weights (~1.7 GB f32).

The one known source of last-digit drift: Python computes the calibrated
softmax in float32 (numpy), the port in doubles, so a probability that sits
within ~1e-7 of a 4-decimal rounding boundary can round differently.
