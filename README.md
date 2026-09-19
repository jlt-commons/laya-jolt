# laya-jolt

Pure-Clojure inference for the Laya decision models, running on
[jolt](https://github.com/jolt-lang/jolt) (Chez Scheme, no JVM). Same weights,
same outputs: the stack reproduces the Python package's `Agent.system_one`
answer for the README quickstart byte-for-byte (under Apple's Accelerate;
within one unit in the fourth decimal under OpenBLAS), on all three
checkpoints of the [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
bundle, and picks the checkpoint per request the way the package's `Router`
does.

The models are ModernBERT encoders (RoPE, alternating full/sliding
attention, GeGLU; ModernBERT-large for `english` and `typed-decisions`,
mmBERT-base for `multilingual`) plus a 2-layer decision head, a scorer, and
an act head. They do not generate text: they consume a serialized `state`
and a set of typed questions, and return calibrated typed answers.

Everything is f32 end to end. F16 checkpoint weights are widened to f32 once,
during `prepare`, so the numerics match the torch CPU oracle exactly.

## Getting the checkpoints

The weights are not in this repo and not in the GitHub `laya` repo either
(that one is the Python package). They live on the Hugging Face Hub:
**https://huggingface.co/convaiinnovations/laya**, one repo bundling three
checkpoints:

| name | where in the repo | encoder | context | for |
|---|---|---|---|---|
| `english` | the root | ModernBERT-large, 421M | 512 | English text |
| `typed-decisions` | `typed-decisions/` | ModernBERT-large, 421M | 1024 | the four typed-decisions workflows (invoice processing, security incidents, customer service, agent-trace observability) |
| `multilingual` | `multilingual/` | mmBERT-base, 322M | 1024 | 100+ languages (Gemma sentencepiece tokenizer, 256k vocab) |

`jolt prepare` reads five files per checkpoint from a directory laid out
like the repo: the root is english, the subfolders are optional.

```
../laya/
  model.safetensors          # ~800 MB, F16
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  encoder/config.json
  rl_agent_config.json
  typed-decisions/           # same five files, optional (~800 MB)
  multilingual/              # same five files, optional (~640 MB)
```

Fetch them with nothing but curl (trim the first loop to the checkpoints
you want; `""` is english):

```
for sub in "" typed-decisions/ multilingual/; do
  mkdir -p ../laya/${sub}tokenizer ../laya/${sub}encoder
  for f in model.safetensors tokenizer/tokenizer.json tokenizer/tokenizer_config.json encoder/config.json rl_agent_config.json; do
    curl -fL -o ../laya/$sub$f https://huggingface.co/convaiinnovations/laya/resolve/main/$sub$f
  done
done
```

or clone the whole model repo with git-lfs (`git lfs install && git clone
https://huggingface.co/convaiinnovations/laya ../laya`), or with the Hub CLI
(`hf download convaiinnovations/laya --local-dir ../laya`). Put it anywhere
and point `LAYA_HOME` at it (or `jolt -M:prepare --laya DIR --out data`).
`jolt prepare` converts every checkpoint it finds there into the same layout
under the data root (`data/`, `data/typed-decisions`, `data/multilingual`);
`--model NAME` converts one. It refuses a root that lacks any of the five
files and says so.

## Build and run

```
jolt kernels             # compile native/laya_kernels.c
jolt prepare             # every checkpoint under ../laya -> data/, data/typed-decisions, ...
jolt -M:test             # parity suites vs golden/
jolt -M:run demo         # README quickstart through the workflow runner
jolt -M:serve            # HTTP API on http://127.0.0.1:8080
jolt binary              # standalone ./laya-server, self-tested against golden/
```

`jolt kernels` shells out to `cc`. `jolt prepare` needs only the checkpoints
and the kernel library; it runs in a few seconds per checkpoint. No Python
is involved anywhere; `golden/` holds the traces dumped from the torch CPU
oracle (english at the root, `golden/typed-decisions/` for that checkpoint)
and is checked in.

`jolt -M:run demo` prints the quickstart answer JSON. It should be identical
to the `:system-one` value in `golden/readme.edn`.

Answers come back as ordered maps with string keys, in the shape of the
Python dicts. Because option order and question order are part of the model
input, pass `:criteria` and the questions map as ordered maps (`array-map`,
or a literal with at most 8 entries); a hash-map would reorder them.

```clojure
(require '[laya.agent :as ag] '[laya.workflows :as wf])
(def agent (ag/load-agent "data"))
(def email (wf/load-workflow "workflows/email.clj"))          ; or (wf/load-workflows dirs)
(ag/system-one agent
               (wf/state email {"subject" "Duplicate billing" "body" raw-body "from" "customer@acme.com"})
               (wf/questions email))
```

## Configuration: `~/.config/laya`

Every entry point (`jolt prepare`, `jolt -M:run`, `jolt -M:serve`, the
binary) resolves its settings the same way: **CLI flag > environment
variable > `config.edn` > default**. `config.edn` lives in
`$LAYA_CONFIG_DIR`, else `$XDG_CONFIG_HOME/laya`, else `~/.config/laya`:

```clojure
{:data "/Users/me/models/laya-data"      ; prepared model dir: what jolt prepare writes and everything else loads
 :laya-home "/Users/me/models/laya"      ; the Hub checkpoint jolt prepare reads
 :workflow-dirs ["/Users/me/src/decisions/workflows"]   ; extra workflow directories
 :port 8080 :host "127.0.0.1" :api-key "s3cret"}       ; server defaults
```

| setting | flag | environment | `config.edn` | default |
|---|---|---|---|---|
| prepared model | `--data DIR` | `LAYA_DATA` | `:data` | `data` |
| checkpoint (prepare) | `--laya DIR` | `LAYA_HOME` | `:laya-home` | `../laya` |
| workflow dirs | `--workflows DIR[:DIR]` | `LAYA_WORKFLOWS` | `:workflow-dirs` (adds) | see below |
| server | `--port` `--host` `--api-key` | `PORT` `LAYA_HOST` `LAYA_API_KEY` | `:port` `:host` `:api-key` | `8080` `127.0.0.1` none |

`--workflows` and `LAYA_WORKFLOWS` are the exception to "adds": they name
exactly the directories to scan, replacing the defaults, so a test or a
one-off run is isolated from whatever is in `~/.config/laya`.

## Workflows

A workflow packages a use case: how to turn raw input into the model's
state, and which typed questions to ask. They are ordinary Clojure files,
not part of `src/`: the bundled ones live in [`workflows/`](workflows) and
yours go in `~/.config/laya/workflows/` (or any directory listed in
`config.edn :workflow-dirs`). Directories load in that order and a later
one wins on a name clash, so a `~/.config/laya/workflows/email.clj`
replaces the bundled `email`.

A file `<dir>/<name>.clj` defines the namespace `workflows.<name>`
(underscores in the file name become dashes) with:

```clojure
(ns workflows.refund-risk
  (:require [clojure.string :as str]))

(defn questions
  "Refund risk on a support ticket."                 ; the docstring is the description
  ([] (questions {}))
  ([opts]                                            ; optional 1-arity: the caller's options
   {"wants_refund" {"type" "noul" "instructions" "Does the customer ask for money back?"}
    "tone" {"type" "score" "instructions" "How angry is the ticket?"
            "criteria" ["calm" "annoyed" "furious"]}
    "team" {"type" "choice" "instructions" "Who should own this?"
            "criteria" (get opts "teams" {"billing" "money" "support" "everything else"})}}))

(defn state                                          ; optional; without it the input is the state
  [input]
  {"ticket" (str/trim (get input "text" ""))})
```

`questions` is required; `state` is optional. Question maps use string keys
(the shape the HTTP API receives); an `array-map` keeps option order, which is
model input. Run it:

```
jolt -M:run --list                                          # what is loaded, from where
jolt -M:run refund-risk '{"text": "Charged twice, want my money back"}'
jolt -M:run refund-risk @ticket.json --options '{"teams": {"billing": "money", "fraud": "chargebacks"}}'
LAYA_WORKFLOWS=./my-workflows jolt -M:run refund-risk @ticket.json   # only that directory
```

Bundled, all byte-identical to the Python package's presets
(`golden/presets.edn`):

| workflow | input | asks |
|---|---|---|
| `demo` | nothing (the quickstart email; pinned by `golden/readme.edn`) | department, urgency, churn risk, phishing |
| `email` | `{"subject", "body", "from"}`; cleans quoted history, signatures, disclaimers | category (option `{"categories" {key description}}` swaps the teams), spam, phishing, urgency, needs reply |
| `triage` | `{"message"}` or a string | intent, urgency, frustration, refund requested, churn risk |
| `guard` | `{"prompt"}` or a string | jailbreak, prompt injection, sensitive data, harm severity, topic |
| `moderation` | `{"post"}` or a string | toxic, harassment, threat, spam, severity |
| `llm-router` | `{"request"}` or a string | difficulty, domain, needs tools, is sensitive (routing *your* LLM traffic; `laya.router` picks Laya checkpoints) |

## HTTP API

`laya.server` mirrors the [TypeSafe Jev API](https://docs.typesafe.ai/api)
and adds the Python package's `Router` and presets. Routes are dispatched by
[ruuter](https://github.com/askonomm/ruuter).

```
POST /v1/systemone        Authorization: Bearer <key>   (only if a key is configured)
{"state": <string|object|array>, "questions": {"<id>": {...}}, "model"?: ..., "lang"?: ..., "task"?: ...}
-> {"model": "laya-rl-agent", "answers": {"<id>": {...}}, "usage": {"input_tokens": n, "output_tokens": 0},
    "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
                "detection": {...}, "workflow": null}}

POST /v1/route            same body, questions optional -> the routing decision alone (nothing loaded or run)
POST /v1/workflows/<name> {"input": <anything the workflow's state fn takes>, "options"?: {...}, "model"?/"lang"?/"task"?}
                          -> the systemone answer + "workflow" + the built "state"
GET  /v1/models           -> {"default": ..., "max_loaded": n, "models": {"english": {"repo", "data", "available", "loaded"}, ...}}
GET  /v1/workflows        -> {"workflows": {"email": {"description", "file", "questions": [ids], "options": bool}, ...}}
GET  /health              -> {"status": "ok", "model": "laya-rl-agent", "loaded": [...], "workflows": [...]}
```

Questions and answers have the shapes the Python `Agent.system_one` uses
(choice / score / noul, plus the `action.act_probability` extension).
`model` is either absent (or `laya-rl-agent`) to route by content, or a
checkpoint name / alias (`english`, `multilingual`, `typed-decisions`, `en`,
`ml`, ...) to pick one; `lang` (`"de"`, `"en-GB"`) and `task`
(`"typed_decisions"`) are the Router's other hints, in the same precedence
as upstream: model > task > detected workflow (opt-in) > lang > detected
script/language > default. Every answer says what was chosen and why under
`routing`. Errors: `401` for a missing or wrong key, `422` with
`{"detail": [{"loc": ["body", "questions", "<id>", "criteria"], "msg": ..., "type": ...}]}`
for anything wrong with the body (malformed JSON, missing state or
questions, unknown type or model, criteria that don't fit the type or the
head), `404` for unknown routes and workflows, `405` for the wrong method,
`503` when the chosen checkpoint has no prepared data, `413` past
`:max-request-bytes` (4 MiB). Inference and checkpoint loading are
serialized on one lock; the adapter's workers overlap only on I/O.

```
jolt -M:serve --port 8080 --host 0.0.0.0 --api-key s3cret   # or PORT / LAYA_HOST / LAYA_API_KEY / LAYA_DATA, or config.edn
curl -s -H 'Authorization: Bearer s3cret' -H 'Content-Type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days.",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  http://127.0.0.1:8080/v1/systemone
curl -s -H 'Authorization: Bearer s3cret' -d '{"input": {"subject": "Refund", "body": "Charged twice.\n\nThanks,\nBob"}}' \
  http://127.0.0.1:8080/v1/workflows/email
```

The server holds one data root (`--data`, the layout `jolt prepare`
writes: `DIR/` english, `DIR/multilingual`, `DIR/typed-decisions`), loads
the default checkpoint at startup and the others on first use, keeping
`--max-loaded` (default 1) resident with least-recently-used eviction: all
three together are ~4.6 GB of f32. A request routed to a checkpoint that
was never prepared gets a `503` saying so.

### As a library

Add this repo as a `:git/url` dep, run `jolt kernels` / `jolt prepare` for
the native library and `data/`, then:

```clojure
(require '[laya.agent :as ag] '[laya.server :as server] '[laya.config :as cfg])
(def agent (ag/load-agent (cfg/setting (cfg/context {}) "--data" "LAYA_DATA" :data "data")))
(ag/system-one agent state questions)                  ; the Python API, as data (~1.7 GB f32, loaded once)
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

Tagged releases (`v*`) carry this binary prebuilt for macOS arm64 and Linux
x86_64, built and self-tested by `.github/workflows/release.yml`. CI runs the
suite on both platforms on every push, fetching the checkpoint from the Hub at
the revision `golden/` was dumped from (`.github/actions/setup`).

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

Encoder, head, both tokenizers, sequence, agent, the workflows and the
checkpoint conversion all match their golden traces on all three
checkpoints (`golden/`, `golden/typed-decisions/`, `golden/multilingual/`).
`system-one` on the quickstart case is byte-identical to the Python output
on each of them under Accelerate; language detection, the Router's
decisions and the presets match the Python package on every pinned case.

Per-forward temporaries live in an ffi arena that closes with the call, so a
long-running process stays at the size of the loaded weights (~1.7 GB f32
per ModernBERT-large checkpoint, 1.3 GB for mmBERT-base; the server keeps
`--max-loaded` of them).

Two known sources of last-digit drift, both one unit in the fourth decimal
of a probability sitting on a rounding boundary: Python computes the
calibrated softmax in float32 (numpy) and the port in doubles; and sgemm
summation order differs between BLAS libraries, and between the CPU
models OpenBLAS picks its kernels for (the same weights gave 0.3142 on one
Linux CI machine and 0.3143 on another). The suite asserts byte identity
under Accelerate and the one-ulp bound everywhere.
