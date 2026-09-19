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

## Build and run

```
jolt kernels             # compile native/laya_kernels.c
jolt prepare             # ../laya checkpoint -> data/   (LAYA_HOME to point elsewhere)
jolt -M:test             # parity suites vs golden/
jolt -M:run              # README quickstart demo
jolt -M:serve            # HTTP API on http://127.0.0.1:8080
jolt binary              # standalone ./laya-server, self-tested against golden/
```

`jolt kernels` shells out to `cc`. `jolt prepare` needs only the checkpoint
(`LAYA_HOME`, default `../laya`) and the kernel library; it runs in a few
seconds. No Python is involved anywhere; `golden/` holds the traces dumped
from the torch CPU oracle and is checked in.

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

## HTTP API

`laya.server` mirrors the [TypeSafe Jev API](https://docs.typesafe.ai/api):

```
POST /v1/systemone        Authorization: Bearer <key>   (only if a key is configured)
{"state": <string|object|array>, "model": "rl-agent", "questions": {"<id>": {...}}}
-> {"model": ..., "answers": {"<id>": {...}}, "usage": {"input_tokens": n, "output_tokens": 0}}

GET  /health              -> {"status": "ok", "model": "rl-agent"}
```

Questions and answers have the shapes the Python `RLAgent.system_one`
uses (choice / score / noul, plus the `rl_agent.act_probability` extension).
`model` is optional and echoed back; it defaults to `rl-agent`. Errors:
`401` for a missing or wrong key, `422` with
`{"detail": [{"loc": ["body", "questions", "<id>", "criteria"], "msg": ..., "type": ...}]}`
for anything wrong with the body (malformed JSON, missing state or
questions, unknown type, criteria that don't fit the type or the head),
`404`/`405` elsewhere, `413` past `:max-request-bytes` (4 MiB). Inference is
serialized on one lock; the adapter's workers overlap only on I/O.

```
jolt -M:serve --port 8080 --host 0.0.0.0 --api-key s3cret   # or PORT / LAYA_HOST / LAYA_API_KEY / LAYA_DATA
curl -s -H 'Authorization: Bearer s3cret' -H 'Content-Type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days.",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  http://127.0.0.1:8080/v1/systemone
```

### As a library

Add this repo as a `:git/url` dep, run `jolt kernels` / `jolt prepare` for
the native library and `data/`, then:

```clojure
(require '[laya.agent :as ag] '[laya.server :as server])
(def agent (ag/load-agent "data"))                     ; ~1.7 GB of f32 weights, once
(ag/system-one agent state questions)                  ; the Python API, as data
(def h (server/handler agent {:api-key nil}))          ; a ring handler to mount anywhere
(def s (server/start agent {:port 8080}))              ; or run it on ring-chez-adapter
(server/stop s)
```

### As a binary

`jolt binary` runs `jolt build -m laya.server -o laya-server` with the C
kernels linked in statically, then runs `./laya-server --self-test` against
`golden/`. The suite runs interpreted, and jolt 0.8.9's release build
miscompiles one pattern (a `reduce` whose accumulator starts as `nil` and is
tested with `nil?` — see `laya.tokenizer/lowest-ranked-pair`), so the binary
proves itself before it ships. It still needs `data/` next to it (or
`--data DIR`), ICU and BLAS from the OS, and libssl/libcrypto for the
adapter.

```
./laya-server --data data --port 8080 --api-key s3cret
./laya-server --self-test --data data --golden golden
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
