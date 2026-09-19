# lev

System One decisions — typed questions over a state, answered with
calibrated probabilities — as one Clojure binary on
[jolt](https://github.com/jolt-lang/jolt) (Chez Scheme, no JVM), with two
kinds of model behind the same API:

- **the encoders**: ModernBERT-large / mmBERT-base with a decision head
  (the `convaiinnovations/laya` checkpoints, three of them), one forward
  pass for all of a call's questions, ~100 ms on a laptop CPU, calibrated
  probabilities, the checkpoints' own Python package reproduced to the
  fourth decimal;
- **a thinker**: any GGUF chat model through llama.cpp, linked into the
  binary. It reads the state and the question, thinks, and its candidate
  answers are scored by their token log probabilities. Seconds a
  question, and right where the encoders are not: on the adversarial
  authored144 set (`bench/`) MiniCPM5-2B answers 97% with thinking
  against the encoders' 61–76%.

A request names the model (`"model": "english"` or `"minicpm5"`) or is
routed by content to an encoder; a confidence gate can answer on the
encoder and escalate only what it is unsure about to the thinker. Either
kind can be configured alone: a box with only encoders serves them, a box
with only a thinker routes everything to it. Constraints tie a call's
questions together; workflows package a use case; von's composable
patterns (route, composite score, two-stage choice) and its
`decide`/`judge`/`rate` are there as a library and over HTTP.

Which one, and when — measured on a 120-case in-distribution set (AG
News / BoolQ / SST-5, the trio localjev's bake-off uses) and on von's
adversarial authored144 (details in [bench/](bench/README.md)):

| | AG News | BoolQ | SST-5 | authored144 | ms per question |
|---|---|---|---|---|---|
| encoder `english` | **97.5%** | 72.5% | 27.5% | 61% | **125** (CPU) |
| thinker, thinking off | 82.5% | 70.0% | 27.5% | 74% | 152 (GPU) |
| thinker, thinking | 85.0% | **90.0%** | **37.5%** | **95%** | 3,000 (GPU) |

The encoder is the fast path: on routing-style traffic it beats the 2.5B
model answering at once, at a fraction of the cost on a CPU, batching a
whole workflow into one forward, with confidence that means something
(its calibration is fitted; the gate relies on it). The thinker earns
its seconds on the cases that need a deduction or an abstention.

The encoders are f32 end to end: F16 checkpoint weights are widened once,
during `prepare`, so the numerics match the torch CPU oracle. (The project
was `laya-jolt` until 2026-09-19; the namespaces are `lev.*`, the binary
`lev-server`, the config `~/.config/lev`, the environment `LEV_*`. The
checkpoints keep their Hub name; the answers' `"model"` is the name of the
model that answered — `english`, `multilingual`, `typed-decisions` or a
thinker's — where the Python package printed its own.)

## Getting the checkpoints

The weights are not in this repo and not in the Python package's GitHub
repo either. They live on the Hugging Face Hub:
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
and point `LEV_CHECKPOINTS` at it (or `jolt -M:prepare --checkpoints DIR --out data`).
`jolt prepare` converts every checkpoint it finds there into the same layout
under the data root (`data/`, `data/typed-decisions`, `data/multilingual`);
`--model NAME` converts one. It refuses a root that lacks any of the five
files and says so.

### A thinker's model

Any chat GGUF llama.cpp loads. The one measured here is
[openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
(2.5B, Apache-2.0, a thinking mode in its chat template): `Q8_0` is 2.7 GB,
`Q4_K_M` 1.6 GB.

```
hf download openbmb/MiniCPM5-2B-GGUF MiniCPM5-2B-Q8_0.gguf --local-dir ~/models
```

Then name it in `~/.config/lev/config.edn` (`:thinkers`, below) or pass
`--thinker ~/models/MiniCPM5-2B-Q8_0.gguf` / `LEV_THINKER`. The prompt
format is ChatML with MiniCPM's `<think>` switch; another model family
needs its own template (`lev.think/defaults`).

## Build and run

```
jolt kernels             # compile native/lev_kernels.c
jolt llama               # clone + build llama.cpp (pinned tag) into native/liblev_llm.*, the thinker's native
jolt prepare             # every checkpoint under ../laya -> data/, data/typed-decisions, ...
jolt -M:test             # parity suites vs golden/
jolt -M:run demo         # README quickstart through the workflow runner
jolt -M:serve            # HTTP API on http://127.0.0.1:8080
jolt binary              # standalone ./lev-server (kernels + llama.cpp linked in), self-tested against golden/
```

jolt 0.8.10 or newer (`deps.edn :jolt/min-version`; an older runtime
refuses the tree, and CI always installs the latest release). `jolt kernels`
shells out to `cc`; `jolt llama` to `git`, `cmake` and `cc`, and is optional:
without its library the encoders run and only the thinker is missing. `jolt prepare` needs only the checkpoints
and the kernel library; it runs in a few seconds per checkpoint. No Python
is involved anywhere; `golden/` holds the traces dumped from the torch CPU
oracle (english at the root, `golden/typed-decisions/` and
`golden/multilingual/` for the others) and is checked in.

`jolt -M:test` runs everything against whatever is prepared under `data/`;
`jolt -M:test lev.checkpoints-test` runs one namespace, and
`LEV_TEST_CHECKPOINTS=typed-decisions` (comma-separated, empty for none)
restricts the extra-checkpoint parity suite, which is how CI tests one
checkpoint per process. `jolt -M:run demo` prints the quickstart answer
JSON: the `:system-one` value in `golden/readme.edn` (to the fourth
decimal, see Status).

Answers come back as ordered maps with string keys, in the shape of the
Python dicts. Because option order and question order are part of the model
input, pass `:criteria` and the questions map as ordered maps (`array-map`,
or a literal with at most 8 entries); a hash-map would reorder them.

```clojure
(require '[lev.agent :as ag] '[lev.workflows :as wf])
(def agent (ag/load-agent "data"))
(def email (wf/load-workflow "workflows/email.clj"))          ; or (wf/load-workflows dirs)
(ag/system-one agent
               (wf/state email {"subject" "Duplicate billing" "body" raw-body "from" "customer@acme.com"})
               (wf/questions email))
```

## Configuration: `~/.config/lev`

Every entry point (`jolt prepare`, `jolt -M:run`, `jolt -M:serve`, the
binary) resolves its settings the same way: **CLI flag > environment
variable > `config.edn` > default**. `config.edn` lives in
`$LEV_CONFIG_DIR`, else `$XDG_CONFIG_HOME/lev`, else `~/.config/lev`:

```clojure
{:data "/Users/me/models/lev-data"       ; prepared data root: what jolt prepare writes and everything else loads
 :checkpoints-home "/Users/me/models/laya"  ; the Hub checkpoints jolt prepare reads
 :encoders {"english" "/Users/me/models/lev-data"}   ; prepared encoders by name (else the :data layout); leave one out to not serve it
 :thinkers {"minicpm5" {:model "/Users/me/models/MiniCPM5-2B-Q8_0.gguf"}}   ; generative models (see Thinkers); none = encoders only
 :workflow-dirs ["/Users/me/src/decisions/workflows"]   ; extra workflow directories
 :port 8080 :host "127.0.0.1" :api-key "s3cret"        ; server defaults
 :max-loaded 2 :max-thinkers 1                          ; encoders / thinkers kept resident
 :default-model "english"                               ; loaded at startup; content routing's fallback (an encoder or a thinker)
 :auto-task-detection false                             ; route typed-decisions question sets to that checkpoint
 :max-len 768 :head-max-len 192                         ; sequence limits for every checkpoint (see Context)
 :checkpoints {"multilingual" {:max-len 2048}}}         ; ... and per checkpoint, which wins
```

A server needs at least one model of either kind and serves whatever is
available: with no thinker, `model` names only encoders; with no prepared
encoder and a thinker as `:default-model`, every request (including the
content-routed ones) goes to the thinker; a request for a model that is
configured but not available (no data directory, no GGUF, the llm native
not built) is a 503, and `GET /v1/models` says which is which.

| setting | flag | environment | `config.edn` | default |
|---|---|---|---|---|
| prepared data root | `--data DIR` | `LEV_DATA` | `:data` | `data` |
| checkpoints (prepare) | `--checkpoints DIR` | `LEV_CHECKPOINTS` | `:checkpoints-home` | `../laya` |
| encoders served | — | — | `:encoders {"name" dir}` | the `:data` layout |
| thinkers served | `--thinker PATH.gguf` (as `thinker`) | `LEV_THINKER` | `:thinkers {"name" {...}}` | none |
| workflow dirs | `--workflows DIR[:DIR]` | `LEV_WORKFLOWS` | `:workflow-dirs` (adds) | see below |
| server | `--port` `--host` `--api-key` | `PORT` `LEV_HOST` `LEV_API_KEY` | `:port` `:host` `:api-key` | `8080` `127.0.0.1` none |
| resident encoders / thinkers | `--max-loaded N` / `--max-thinkers N` | `LEV_MAX_LOADED` / `LEV_MAX_THINKERS` | `:max-loaded` / `:max-thinkers` | `1` / `1` |
| startup / fallback model | `--default-model NAME` | `LEV_DEFAULT_MODEL` | `:default-model` | `english` |
| typed-decisions by question ids | `--auto-task-detection` | — | `:auto-task-detection` | off |
| sequence limits | `--max-len N` `--head-max-len N` | `LEV_MAX_LEN` `LEV_HEAD_MAX_LEN` | `:max-len` `:head-max-len`, `:checkpoints {"name" {…}}` | the checkpoint's own (`rl_agent_config.json`) |
| goldens (`--self-test`) | `--golden DIR` | `LEV_GOLDEN` | `:golden` | `golden` |

`--workflows` and `LEV_WORKFLOWS` are the exception to "adds": they name
exactly the directories to scan, replacing the defaults, so a test or a
one-off run is isolated from whatever is in `~/.config/lev`.

### Thinkers

```clojure
{:thinkers {"minicpm5" {:model "/Users/me/models/MiniCPM5-2B-Q8_0.gguf"
                        :thinking true          ; think before answering (a request can override)
                        :max-think-tokens 1024  ; the budget; the thought is closed when it runs out
                        :n-ctx 4096 :n-gpu-layers -1 :threads 0   ; llama.cpp: context, layers on the GPU (-1 all), threads (0 = its default)
                        :temperature 1.0 :top-p 0.95 :min-p 0.0 :seed 42}}   ; sampling of the thought
 :max-thinkers 1}                                ; resident at once (each is GBs)
```

Each entry is a model name a request can ask for. `--thinker PATH` or
`LEV_THINKER` adds one named `thinker`. Thinkers are loaded on first use,
never chosen by content routing, and listed by `GET /v1/models`. Without
the llm native (`jolt llama`) or the GGUF on disk a thinker is listed as
unavailable and a request for it is a 503.

## Context: what the model sees

The encoders have no sessions, turns or memory. Every call is one stateless forward
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
`--max-len` / `--head-max-len`, or `LEV_MAX_LEN` / `LEV_HEAD_MAX_LEN`
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
yours go in `~/.config/lev/workflows/` (or any directory listed in
`config.edn :workflow-dirs`). Directories load in that order and a later
one wins on a name clash, so a `~/.config/lev/workflows/email.clj`
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
LEV_WORKFLOWS=./my-workflows jolt -M:run refund-risk @ticket.json   # only that directory
```

### Constraints

The model answers each question on its own. A workflow (or a request) can
tie them with constraints, decided jointly after the forward pass
(`lev.constraints`, the port of GLiNER2's constrained classification):

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

Bundled: five with questions byte-identical to the Python package's presets
(`golden/presets.edn`; `email` adds constraints the Python package has no
equivalent of) and `security` from von:

| workflow | input | asks |
|---|---|---|
| `demo` | nothing (the quickstart email; pinned by `golden/readme.edn`) | department, urgency, churn risk, phishing |
| `email` | `{"subject", "body", "from"}`; cleans quoted history, signatures, disclaimers | category (option `{"categories" {key description}}` swaps the teams), spam, phishing, urgency, needs reply |
| `triage` | `{"message"}` or a string | intent, urgency, frustration, refund requested, churn risk |
| `guard` | `{"prompt"}` or a string | jailbreak, prompt injection, sensitive data, harm severity, topic |
| `moderation` | `{"post"}` or a string | toxic, harassment, threat, spam, severity |
| `llm-router` | `{"request"}` or a string | difficulty, domain, needs tools, is sensitive (routing *your* LLM traffic; `lev.router` picks lev models) |
| `security` | `{"event"}` or a string | event type, active threat, severity (von's security preset), with constraints: benign is no threat, a threat is at least elevated |

## HTTP API

`lev.server` mirrors the [TypeSafe Jev API](https://docs.typesafe.ai/api)
and adds the Python package's `Router` and presets. Routes are dispatched by
[ruuter](https://github.com/askonomm/ruuter).

```
POST /v1/systemone        Authorization: Bearer <key>   (only if a key is configured)
{"state": <string|object|array>, "questions": {"<id>": {...}}, "constraints"?: [...], "on_infeasible"?: "min_violations"|"raise",
 "model"?: ..., "lang"?: ..., "task"?: ...}
-> {"model": "english", "answers": {"<id>": {...}}, "usage": {"input_tokens": n, "output_tokens": 0},
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
GET  /health              -> {"status": "ok", "model": "lev", "loaded": [...], "thinkers": [...], "workflows": [...]}
```

Questions and answers have the shapes the Python `Agent.system_one` uses
(choice / score / noul, plus the `action.act_probability` extension on the
encoders' answers). A thinker answers in the same shapes without `action`,
with a `"thinking": {"enabled", "tokens", "max_tokens"}` report and, on
request (`"thought": true`), each answer's reasoning under `"thought"`;
`"thinking": false` asks it to answer at once (~150 ms on a GPU, 74% on
authored144 against 97% with thinking).

```
POST /v1/systemone {"state": ..., "questions": {...}, "model": "minicpm5", "thinking": true, "thought": false}
POST /v1/systemone {"state": ..., "questions": {...}, "escalate": {"model": "minicpm5", "threshold": 0.8, "thinking": true}}
                   -> the encoder's answers, the ones below the threshold replaced by the thinker's, plus
                      "escalation": {"threshold", "model", "escalated": [ids], "usage": the thinker's}
POST /v1/patterns/confidence-gate   systemone body + "threshold"          -> {"automatic": {...}, "escalate": {...}, "response": {...}}
POST /v1/patterns/composite-score   systemone body + "weights" {id: w}    -> {"score": 0..1, "breakdown": {...}, "response": {...}}
POST /v1/patterns/two-stage-choice  {"state", "taxonomy": {category: {option: description}}} -> {"category", "choice", "combined_confidence", ...}
```

`escalate` also works on a workflow request; constraints then decide over
the merged answers.
`model` is either absent (or `lev`, or a TypeSafe SDK's default
`jev-latest` / `jev-preview`) to route by content, a checkpoint name /
alias (`english`, `multilingual`, `typed-decisions`, `en`, `ml`, ...) or a
thinker's name to pick one; `lang` (`"de"`, `"en-GB"`) and `task`
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
jolt -M:serve --port 8080 --host 0.0.0.0 --api-key s3cret   # or PORT / LEV_HOST / LEV_API_KEY / LEV_DATA, or config.edn
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
(require '[lev.agent :as ag] '[lev.server :as server] '[lev.config :as cfg])
(def agent (ag/load-agent (cfg/setting (cfg/context {}) "--data" "LEV_DATA" :data "data")))
(ag/system-one agent state questions)                  ; the Python API, as data (~1.7 GB f32, loaded once)
(def h (server/handler agent {:api-key nil}))          ; a ring handler to mount anywhere
(def s (server/start agent {:port 8080}))              ; or run it on ring-chez-adapter
(server/stop s)
```

Or route between the checkpoints like the Python `Router`, loading each on
first use and keeping `:max-loaded` resident:

```clojure
(require '[lev.router :as router])
(def rt (router/make-router {:data "data" :max-loaded 2}))          ; data/, data/multilingual, data/typed-decisions
(router/route rt {"body" "Der Kunde wurde zweimal belastet"} questions)   ; the decision, nothing loaded
;; => {"model" "multilingual", "repo" "convaiinnovations/laya/multilingual",
;;     "reason" "Latin script but language looks like 'de', not English", ...}
(router/predict rt state questions)                                  ; system-one + "routing"
(router/predict rt state questions :model "typed-decisions")         ; or :lang "de", :task "typed_decisions"
(router/loaded rt)                                                   ; ["multilingual" "typed-decisions"]
(def h (server/handler rt {:workflows (lev.workflows/load-workflows ["workflows"])}))
```

The one-question conveniences and the patterns, as a library (`lev.api`,
`lev.patterns`; `agent` is an agent or a router):

```clojure
(require '[lev.api :as api] '[lev.patterns :as pat] '[lev.router :as router])
(def rt (router/make-router {:data "data" :thinkers {"minicpm5" {:model "/Users/me/models/MiniCPM5-2B-Q8_0.gguf"}}}))

(api/decide rt "Database replication lag exceeded 45 seconds." {"infrastructure" "servers, network" "billing" "invoices"})
;; => {"type" "choice" "choice" "infrastructure" "probabilities" {...} "confidence" 0.83 "action" {...}}
(api/judge rt "Connection pool exhausted; handshakes timing out." "Is this blocking customers?")   ; => 0.9412
(api/rate rt "Memory at 98%, OOM killer active." ["nominal" "degraded" "critical"])                ; => the score answer
(api/decide rt state choices instructions {:model "minicpm5" :thinking true})                      ; the thinker

(pat/confidence-gate rt state questions {:threshold 0.85})           ; {"automatic" .. "escalate" .. "response" ..}
(pat/escalate rt state questions {:threshold 0.8 :model "minicpm5"}) ; the gate, with the escalated questions re-asked
(pat/route rt event (api/choice "Dispute action?" {"refund" "" "escalate" ""}) {"refund" process-refund "escalate" notify-fraud} {})
(pat/composite-score rt telemetry questions {:weights {"severity" 2.0 "is_threat" 3.0}})   ; {"score" 0.91 ...}
(pat/two-stage-choice rt "Postgres replica lag" {"cloud" {"aws" "..." "gcp" "..."} "database" {"postgres" "..." "redis" "..."}} {})
```

### As a binary

`jolt binary` runs `jolt build -m lev.server -o lev-server` with the C
kernels and llama.cpp (`jolt llama`: the pinned tag, static, Metal on mac)
linked in, then runs `./lev-server --self-test` against `golden/`; with a
thinker configured the self-test also asks it one question, which proves
the link. The suite runs interpreted, and a compiler release can build
the tree wrong where the interpreter runs it right (jolt 0.8.9 miscompiled a
`reduce` whose accumulator starts as `nil` and is tested with `nil?`, the
pattern `lev.tokenizer/lowest-ranked-pair` uses; 0.8.10 fixed it), so the
binary proves itself before it ships. It still needs the prepared data root next
to it (or `--data DIR` / `config.edn`), the workflows (`./workflows` relative
to where it runs, `--workflows`, `LEV_WORKFLOWS` or `config.edn`, plus
`~/.config/lev/workflows`; the workflow files are loaded from source at
startup, so they need no rebuild), ICU and BLAS from the OS, and
libssl/libcrypto for the adapter.

```
./lev-server --data data --workflows workflows --port 8080 --api-key s3cret
./lev-server --self-test --data data --golden golden
```

Tagged releases (`v*`) carry this binary prebuilt for macOS arm64 and Linux
x86_64, with `golden/` and `workflows/` alongside, built and self-tested by
`.github/workflows/release.yml`. CI runs the suite on both platforms on every
push, fetching the three checkpoints from the Hub at the revision `golden/`
was dumped from (`.github/actions/setup`).

## Speed or accuracy

`bench/` runs any configured model on von's authored144 set (144
adversarial three-way decisions):

```
jolt -M bench/authored144.clj                                   # english: 61%, ~120 ms a case
jolt -M bench/authored144.clj --model typed-decisions            # 67%
jolt -M bench/authored144.clj --model minicpm5 --thinking false  # 74%, ~150 ms a case (Metal)
jolt -M bench/authored144.clj --model minicpm5                   # 97%, seconds a case
```

The gap to a hosted generative decision API is the reasoning budget, not
the encoder: the same 2.5B model answering at once is no better than an
NLI encoder (von-1.0, 76%). Pick per request: `model` names the encoder or
the thinker, `thinking` the budget, `escalate` the gate that spends it
only on the unsure answers. Numbers, alternatives and the method are in
[bench/README.md](bench/README.md).

## Native dependencies

Both platforms are supported; `deps.edn` carries darwin and linux entries and
the build task branches on OS.

- **kernels** — `native/liblev_kernels.dylib` (mac) / `.so` (linux), built by
  `jolt kernels`: the elementwise and reduction kernels and attention, with
  a pthread pool of their own (`LEV_THREADS`). Attention calls
  `cblas_sgemm` through a pointer the Clojure side hands it, so the library
  links against no BLAS.
- **llama.cpp** — `native/liblev_llm.dylib` / `.so` and `liblev_llm.a`,
  built by `jolt llama`: llama.cpp cloned at its pinned tag into
  `native/llama.cpp` and built static (`cmake`; Metal with the shader
  library embedded on mac, CPU elsewhere) behind `native/lev_llm.c`, a
  flat C face jolt.ffi binds (`lev.llm`). Optional: without it the
  encoders run and thinkers are unavailable. For `jolt build` the archive
  is force-loaded and libc++ and the Metal, Foundation, MetalKit and
  Accelerate frameworks are linked through `lib<Name>.tbd` symlinks the
  build script makes under `native/frameworks/` (how a `deps.edn`
  `:static {:lib}` can name a framework); the binary depends only on
  system frameworks.
- **ICU** — `libicucore.dylib` on mac (unguarded symbols in the system dylib),
  `libicuuc.so.<ver>` on linux. The tokenizers call `unorm2` (NFC) and the
  `u_charType` / `u_isUWhiteSpace` classifiers; language detection and the
  email workflow use the same classifiers for Python's `str.isalpha` /
  `\\w` / `\\s`. Linux ICU builds append the
  major version to every symbol (`u_charType_76`); the bindings resolve the
  first spelling that exists, for versions 60..90.
- **JSON** — `org.clojure/data.json` from Maven, plus `jolt-lang/time` which
  provides the `java.time` classes data.json needs to load. Only `prepare`
  uses them; requests are read by `lev.json`, which keeps key order.
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
one question is ~95 ms at 55 tokens and ~350 ms at 512, linear in
between (it was quadratic — 4.8 s at 512 — until attention went through
per-head sgemm with the sliding layers scoring only their 129-key band;
the softmax, LayerNorm and GELU kernels are vectorized and a call's
questions share one batched forward);
the bundled `demo` (4 questions, ~90 tokens each) takes ~290 ms and
`email` on a 3,000-character body (5 questions at the 512 cap) ~1.5 s. The
matmuls are Accelerate's (multi-core); attention (heads x query blocks),
the swiglu GELU and LayerNorm (rows) run on the kernel library's own
thread pool, `LEV_THREADS` wide (default: the online processors;
`lev.tensors/set-threads!` at runtime). The thread count changes only the
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
