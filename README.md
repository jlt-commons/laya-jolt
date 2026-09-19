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
- `src/laya/sequence.clj` — `build_sequence` (question -> markers + ids).
- `src/laya/agent.clj` — `system_one` (temperature calibration, confidence).
- `python/prepare.py` — checkpoint -> `data/` (f32 blobs + EDN manifest).
- `python/dump_traces.py` — dump golden traces from the torch oracle into
  `golden/`. These are the contract the port is tested against.

## Build and run

```
jolt kernels             # compile native/laya_kernels.c
jolt prepare             # ../laya checkpoint -> data/   (needs python3 + LAYA_HOME)
jolt -M:test             # parity suites vs golden/
jolt -M:run              # README quickstart demo
```

`jolt kernels` shells out to `cc`; `jolt prepare` and `jolt traces` need the
checkpoint (`LAYA_HOME`, default `../laya`) and a python3 with `torch`,
`transformers`, and `safetensors` installed.

`jolt -M:run` prints the quickstart answer JSON. It should be identical to the
`:system-one` value in `golden/readme.edn`.

## Native dependencies

Both platforms are supported; `deps.edn` carries darwin and linux entries and
the build task branches on OS.

- **kernels** — `native/liblaya_kernels.dylib` (mac) / `.so` (linux), built by
  `jolt kernels`.
- **NFC** — `libicucore.dylib` on mac (unguarded symbols in the system dylib),
  `libicuuc.so.<ver>` on linux. The tokenizer calls `unorm2`.
- **BLAS** — `cblas_sgemm` from the Accelerate framework on mac, OpenBLAS on
  linux.

On linux, install the ICU and OpenBLAS runtime packages and adjust the version
suffixes in `deps.edn` if your distro's `libicuuc.so` version is not listed.

## Status

Encoder, head, tokenizer, sequence, and agent all match their golden traces.
`system-one` on the quickstart case is byte-identical to the Python output.
