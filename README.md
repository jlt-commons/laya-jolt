# laya-jolt

Pure-Clojure inference for the Laya decision models, running on
[jolt](https://github.com/jolt-lang/jolt) (Chez Scheme, no JVM). Same weights,
same outputs: the stack reproduces the Python package's `Agent.system_one`
answer for the README quickstart to the fourth decimal it prints, on all
three checkpoints of the [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
bundle, and picks the checkpoint per request the way the package's `Router`
does.

The models are ModernBERT encoders (RoPE, alternating full/sliding
attention, a GELU-gated MLP; ModernBERT-large for `english` and `typed-decisions`,
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
oracle (english at the root, `golden/typed-decisions/` and
`golden/multilingual/` for the others) and is checked in.

`jolt -M:test` runs everything against whatever is prepared under `data/`;
`jolt -M:test laya.checkpoints-test` runs one namespace, and
`LAYA_CHECKPOINTS=typed-decisions` (comma-separated, empty for none)
restricts the extra-checkpoint parity suite, which is how CI tests one
checkpoint per process. `jolt -M:run demo` prints the quickstart answer
JSON: the `:system-one` value in `golden/readme.edn` (to the fourth
decimal, see Status).

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
{:data "/Users/me/models/laya-data"      ; prepared data root: what jolt prepare writes and everything else loads
 :laya-home "/Users/me/models/laya"      ; the Hub checkpoints jolt prepare reads
 :workflow-dirs ["/Users/me/src/decisions/workflows"]   ; extra workflow directories
 :port 8080 :host "127.0.0.1" :api-key "s3cret"        ; server defaults
 :max-loaded 2 :default-model "english"                 ; checkpoints kept resident; the one loaded at startup
 :auto-task-detection false                             ; route typed-decisions question sets to that checkpoint
 :max-len 768 :head-max-len 192                         ; sequence limits for every checkpoint (see Context)
 :checkpoints {"multilingual" {:max-len 2048}}}         ; ... and per checkpoint, which wins
```

| setting | flag | environment | `config.edn` | default |
|---|---|---|---|---|
| prepared data root | `--data DIR` | `LAYA_DATA` | `:data` | `data` |
| checkpoints (prepare) | `--laya DIR` | `LAYA_HOME` | `:laya-home` | `../laya` |
| workflow dirs | `--workflows DIR[:DIR]` | `LAYA_WORKFLOWS` | `:workflow-dirs` (adds) | see below |
| server | `--port` `--host` `--api-key` | `PORT` `LAYA_HOST` `LAYA_API_KEY` | `:port` `:host` `:api-key` | `8080` `127.0.0.1` none |
| resident checkpoints | `--max-loaded N` | `LAYA_MAX_LOADED` | `:max-loaded` | `1` |
| startup / fallback checkpoint | `--default-model NAME` | `LAYA_DEFAULT_MODEL` | `:default-model` | `english` |
| typed-decisions by question ids | `--auto-task-detection` | — | `:auto-task-detection` | off |
| sequence limits | `--max-len N` `--head-max-len N` | `LAYA_MAX_LEN` `LAYA_HEAD_MAX_LEN` | `:max-len` `:head-max-len`, `:checkpoints {"name" {…}}` | the checkpoint's own (`rl_agent_config.json`) |
| goldens (`--self-test`) | `--golden DIR` | `LAYA_GOLDEN` | `:golden` | `golden` |

`--workflows` and `LAYA_WORKFLOWS` are the exception to "adds": they name
exactly the directories to scan, replacing the defaults, so a test or a
one-off run is isolated from whatever is in `~/.config/laya`.

## Context: what the model sees

Laya has no sessions, turns or memory. Every call is one stateless forward
pass, and the context is exactly the `state` you pass; the server caches
loaded weights, nothing else. For each question, `build-sequence` lays out

```
[CLS] <type> question: <instructions> [SEP] [MASK] option 0 [MASK] option 1 … [SEP] <state> [SEP]
```

and the encoder reads it bidirectionally in one pass; the head scores the
`[MASK]` marker of each option. The questions in a call share the state but
are independent rows: nothing passes between them, and nothing survives the
call.

**Budget.** `max_len` is 512 tokens on `english`, 1024 on `typed-decisions`
and `multilingual`, as trained. Instructions and options get up to
`head_max_len` (192 / 256; long option lists are shrunk evenly, then the
instructions), the state gets the rest. Measured on the bundled workflows:
427–478 state tokens per question on `english` (roughly 1,700–1,900
characters of English prose, less for JSON), 940–990 on the other two. What
does not fit is dropped from the **end**, silently: the first tokens of the
serialized state are kept. That is why the `email` workflow strips quoted
history, signatures and disclaimers and caps the body at 3,000 characters
before the model sees anything. `usage.input_tokens` in every answer is the
total over all questions, so it tells you when a state is being cut.

Both limits are yours to change: `:max-len` / `:head-max-len` in
`config.edn` (for every checkpoint, or per name under `:checkpoints`),
`--max-len` / `--head-max-len`, or `LAYA_MAX_LEN` / `LAYA_HEAD_MAX_LEN`
(Configuration). They apply when a checkpoint loads; `GET /v1/models`
reports the effective values and the server log says `max_len 768 (trained
512)`. RoPE has no position table, so a longer sequence runs fine
mechanically and simply reads more of the state; the checkpoints were
trained at 512 / 1024, and answer quality past that is unmeasured, so raise
it deliberately and check on your own data. Lowering `head_max_len` buys
state room at the cost of shrinking long option texts sooner.

**Shaping the state.** A string is tokenized as is; a map or vector is
serialized like Python's `json.dumps` (key order kept, `ensure_ascii` off)
and the keys are tokens too. Name them and refer to them in the
instructions with backticks, the way the presets do:

```clojure
(ag/system-one agent
  {"ticket" "Charged twice, want my money back" "plan" "pro" "account_age_days" 412}
  {"churn_risk" {"type" "noul" "instructions" "Does `ticket` suggest the customer may cancel?"}})
```

Put the facts the decision needs in the state, and only those: a short,
structured state beats a long raw one both for the budget and for the
answers.

**Conversations.** The state can be the trajectory so far, a vector of
turns:

```clojure
(def turns [{"role" "user" "text" "My payouts have failed for three days."}
            {"role" "agent" "text" "I see two failed transfers. Can you confirm the account ending 4411?"}
            {"role" "user" "text" "Yes. If this isn't fixed today I'm moving to Stripe."}])
(ag/system-one agent turns
  {"churn_risk" {"type" "noul" "instructions" "Will this customer leave?"}
   "needs_human" {"type" "noul" "instructions" "Should a person take over this conversation?"}})
```

The checkpoints were trained on conversation prefixes with TD(λ = 1)
targets, so re-asking the same questions on the growing prefix after each
turn is the intended use; `rl_agent_config.json` (`max_prefixes 6`) shows
the depth the training used. Your application holds the turns; when they
outgrow the budget, pass the last few or a summary you produce elsewhere.

**Chaining.** Multi-step decisions are successive calls with state you
assemble: `guard` before anything else, `llm-router` to pick a model, then
the workflow for the request. `action.act_probability` in every answer is
the act head's estimate that the system should act rather than escalate,
which is the natural input to gating between calls. A server-side session
store (append a turn, re-run a workflow on the trajectory) is an
application-layer feature the upstream package does not have either; the
workflow contract is where it would go.

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
jolt -M:run refund-risk @ticket.json --constraints '[["implies", ["wants_refund", true], ["team", "billing"]]]'
LAYA_WORKFLOWS=./my-workflows jolt -M:run refund-risk @ticket.json   # only that directory
```

### Constraints

The model answers each question on its own. A workflow (or a request) can
tie them with constraints, decided jointly after the forward pass
(`laya.constraints`, the port of GLiNER2's constrained classification):

```clojure
(defn constraints                                    ; optional; same arities as questions
  []
  [[:implies ["wants_refund" true] ["team" "billing"]]
   [:at-most 1 ["tone" 2] ["team" "support"]]
   [:min-level "tone" 1]])
```

A ref `[question label]` names a choice option, a score level (index or
legend text) or a noul boolean; operators are `not`, `all-of`, `any-of`,
`implies`, `iff`, `excludes`, `exactly-one-of`, `at-least k`, `at-most k`,
`exactly k` (over refs or nested constraints) and, for score questions,
`at-level`, `min-level`, `max-level`, `between-level`. Written as strings,
keywords, dashes or underscores; JSON requests use the same shape.

The decision maximises the joint probability (the sum of the calibrated
log probabilities) subject to the constraints: independent when nothing
couples two questions, else exact search with branch and bound (beam
search past a node budget). When constraints are given, even an empty
list, every answer carries `decided` next to its own field (choice: the
option, score: the level index, noul: the boolean) and the result a
`constraints` report `{"feasible", "decoder", "exact", "violations"}`;
the model's own `choice`, `score`, `noul` and `probabilities` never
change. A contradictory set falls to the assignment with the fewest
violated constraints (listed in `violations`), or fails when
`on_infeasible` is `raise`. The bundled `email` workflow ties `needs_reply`
to `is_spam` and `is_phishing`.

Bundled, their questions byte-identical to the Python package's presets
(`golden/presets.edn`); `email` adds constraints the Python package has no
equivalent of:

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
{"state": <string|object|array>, "questions": {"<id>": {...}}, "constraints"?: [...], "on_infeasible"?: "min_violations"|"raise",
 "model"?: ..., "lang"?: ..., "task"?: ...}
-> {"model": "laya-rl-agent", "answers": {"<id>": {...}}, "usage": {"input_tokens": n, "output_tokens": 0},
    "constraints"?: {"feasible": bool, "decoder": ..., "exact": bool, "violations": [...]},
    "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
                "detection": {...}, "workflow": null}}

POST /v1/route            same body, questions optional -> the routing decision alone (nothing loaded or run)
POST /v1/workflows/<name> {"input": <anything the workflow's state fn takes>, "options"?: {...}, "constraints"?: [...],
                           "on_infeasible"?: ..., "model"?/"lang"?/"task"?}
                          -> the systemone answer + "workflow" + the built "state"; the request's constraints
                             are added to the workflow's own
GET  /v1/models           -> {"default": ..., "max_loaded": n, "models": {"english": {"repo", "data", "available", "loaded"}, ...}}
GET  /v1/workflows        -> {"workflows": {"email": {"description", "file", "questions": [ids], "constraints": [...], "options": bool}, ...}}
GET  /health              -> {"status": "ok", "model": "laya-rl-agent", "loaded": [...], "workflows": [...]}
```

Questions and answers have the shapes the Python `Agent.system_one` uses
(choice / score / noul, plus the `action.act_probability` extension).
`model` is either absent (or the engine's own name, `laya-rl-agent`) to
route by content, or a checkpoint name / alias (`english`, `multilingual`, `typed-decisions`, `en`,
`ml`, ...) to pick one; `lang` (`"de"`, `"en-GB"`) and `task`
(`"typed_decisions"`) are the Router's other hints, in the same precedence
as upstream: model > task > detected workflow (opt-in) > lang > detected
script/language > default. Every answer says what was chosen and why under
`routing`. Errors: `401` for a missing or wrong key, `422` with
`{"detail": [{"loc": ["body", "questions", "<id>", "criteria"], "msg": ..., "type": ...}]}`
for anything wrong with the body (malformed JSON, missing state or
questions, unknown type or model, criteria that don't fit the type or the
head, a constraint that names no question or label — at
`["body", "constraints", i]` — or, with `on_infeasible: raise`, a set
nothing satisfies: type `infeasible`, with the `violations`), `404` for
unknown routes and workflows, `405` for the wrong method,
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

Or route between the checkpoints like the Python `Router`, loading each on
first use and keeping `:max-loaded` resident:

```clojure
(require '[laya.router :as router])
(def rt (router/make-router {:data "data" :max-loaded 2}))          ; data/, data/multilingual, data/typed-decisions
(router/route rt {"body" "Der Kunde wurde zweimal belastet"} questions)   ; the decision, nothing loaded
;; => {"model" "multilingual", "repo" "convaiinnovations/laya/multilingual",
;;     "reason" "Latin script but language looks like 'de', not English", ...}
(router/predict rt state questions)                                  ; system-one + "routing"
(router/predict rt state questions :model "typed-decisions")         ; or :lang "de", :task "typed_decisions"
(router/loaded rt)                                                   ; ["multilingual" "typed-decisions"]
(def h (server/handler rt {:workflows (laya.workflows/load-workflows ["workflows"])}))
```

### As a binary

`jolt binary` runs `jolt build -m laya.server -o laya-server` with the C
kernels linked in statically, then runs `./laya-server --self-test` against
`golden/`. The suite runs interpreted, and a compiler release can build
the tree wrong where the interpreter runs it right (jolt 0.8.9 miscompiled a
`reduce` whose accumulator starts as `nil` and is tested with `nil?`, the
pattern `laya.tokenizer/lowest-ranked-pair` uses; 0.8.10 fixed it), so the
binary proves itself before it ships. It still needs the prepared data root next
to it (or `--data DIR` / `config.edn`), the workflows (`./workflows` relative
to where it runs, `--workflows`, `LAYA_WORKFLOWS` or `config.edn`, plus
`~/.config/laya/workflows`; the workflow files are loaded from source at
startup, so they need no rebuild), ICU and BLAS from the OS, and
libssl/libcrypto for the adapter.

```
./laya-server --data data --workflows workflows --port 8080 --api-key s3cret
./laya-server --self-test --data data --golden golden
```

Tagged releases (`v*`) carry this binary prebuilt for macOS arm64 and Linux
x86_64, with `golden/` and `workflows/` alongside, built and self-tested by
`.github/workflows/release.yml`. CI runs the suite on both platforms on every
push, fetching the three checkpoints from the Hub at the revision `golden/`
was dumped from (`.github/actions/setup`).

## Accuracy

`bench/` runs a prepared checkpoint on von's authored144 set (144
adversarial three-way decisions) and records how the alternatives do on
the same cases: laya `english` 61%, `typed-decisions` 67%, von-1.0 (an NLI
head on the same ModernBERT-large encoder) 76%, a 2.5B decoder answering
directly 74%, the same decoder with ~300 tokens of thinking 97% at 3 s a
case. The gap to a hosted generative decision API is the reasoning budget,
not the encoder; see [bench/README.md](bench/README.md).

## Native dependencies

Both platforms are supported; `deps.edn` carries darwin and linux entries and
the build task branches on OS.

- **kernels** — `native/liblaya_kernels.dylib` (mac) / `.so` (linux), built by
  `jolt kernels`: the elementwise and reduction kernels and attention, with
  a pthread pool of their own (`LAYA_THREADS`). Attention calls
  `cblas_sgemm` through a pointer the Clojure side hands it, so the library
  links against no BLAS.
- **ICU** — `libicucore.dylib` on mac (unguarded symbols in the system dylib),
  `libicuuc.so.<ver>` on linux. The tokenizers call `unorm2` (NFC) and the
  `u_charType` / `u_isUWhiteSpace` classifiers; language detection and the
  email workflow use the same classifiers for Python's `str.isalpha` /
  `\\w` / `\\s`. Linux ICU builds append the
  major version to every symbol (`u_charType_76`); the bindings resolve the
  first spelling that exists, for versions 60..90.
- **JSON** — `org.clojure/data.json` from Maven, plus `jolt-lang/time` which
  provides the `java.time` classes data.json needs to load. Only `prepare`
  uses them; requests are read by `laya.json`, which keeps key order.
- **HTTP** — `jolt-lang/ring-chez-adapter` serves the API and
  `org.clojars.askonomm/ruuter` (Clojars) dispatches its routes.
- **BLAS** — `cblas_sgemm` from the Accelerate framework on mac, OpenBLAS on
  linux.

On linux, install the ICU and OpenBLAS runtime packages and adjust the version
suffixes in `deps.edn` if your distro's `libicuuc.so` version is not listed.

## Status

Encoder, head, both tokenizers, sequence, agent, the workflows and the
checkpoint conversion all match their golden traces on all three
checkpoints (`golden/`, `golden/typed-decisions/`, `golden/multilingual/`).
`system-one` on the quickstart case matches the Python output on each of
them to the four decimals it prints; one value (`urgency` p[1], 0.314250x)
sits on a rounding boundary and prints 0.3142 or 0.3143 depending on the
BLAS and how attention is blocked, so exact bytes are not the contract.
Language detection, the Router's decisions and the presets match the
Python package on every pinned case.

Per-forward temporaries live in an ffi arena that closes with the call, so a
long-running process stays at the size of the loaded weights (~1.7 GB f32
per ModernBERT-large checkpoint, 1.3 GB for mmBERT-base; the server keeps
`--max-loaded` of them). The questions of one call go through the encoder
as one batch of up to 8 rows (padded to the longest, masked), sharing every
matmul; the intermediates are one workspace per call, ~260 MB at 8 x 512.

Speed, `english` on a 10-core M-series laptop, Accelerate, single call:
one question is ~95 ms at 55 tokens and ~350 ms at 512, linear in between;
the bundled `demo` (4 questions, ~90 tokens each) takes ~290 ms and
`email` on a 3,000-character body (5 questions at the 512 cap) ~1.5 s. The
matmuls are Accelerate's (multi-core); attention (heads x query blocks),
the swiglu GELU and LayerNorm (rows) run on the kernel library's own
thread pool, `LAYA_THREADS` wide (default: the online processors;
`laya.tensors/set-threads!` at runtime). The thread count changes only the
schedule, never a result: every task runs the same arithmetic, and the
suite checks the bytes are identical at 1, 2, 3 and 8 threads. On one
thread the same call takes ~520 ms at 512 tokens and `email` ~2.3 s.

Two known sources of last-digit drift, both one unit in the fourth decimal
of a probability sitting on a rounding boundary: Python computes the
calibrated softmax in float32 (numpy) and the port in doubles; and sgemm
summation order differs between BLAS libraries, and between the CPU
models OpenBLAS picks its kernels for (the same weights gave 0.3142 on one
Linux CI machine and 0.3143 on another). The suite asserts the one-unit
bound everywhere.
