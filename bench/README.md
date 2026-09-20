# Benchmarks

`data/authored144.jsonl` is von's authored144 set
([github.com/wfzyx/von](https://github.com/wfzyx/von), Apache-2.0; project-authored,
no copied text): 144 three-way decisions over one- or two-sentence states,
48 each of evidence interpretation (supported / insufficient / contradicted),
rule application (permitted / prohibited / insufficient) and candidate
selection (A / B / neither), labels balanced. `data/perturbations108.jsonl`
re-asks 108 of them with the options reversed or the question rewrapped. The
set is adversarial by design: every case turns on a negation, a scope word or
an abstention, and half the trick is knowing when the evidence does not
settle the question.

```
jolt -M bench/authored144.clj                              # english
jolt -M bench/authored144.clj --model typed-decisions
jolt -M bench/authored144.clj --model minicpm5 --thinking false   # a configured thinker (config.edn :thinkers, or --thinker PATH as `thinker`)
jolt -M bench/authored144.clj --model minicpm5                    # thinking
jolt -M bench/authored144.clj --file bench/data/perturbations108.jsonl --out results.jsonl
```

Each row is one `choice` question: the row's question as the instructions,
its options as the criteria, its state as the state.

## Results, 2026-09-19

10-core M-series laptop. lev through `bench/authored144.clj`; von-1.0
through its own Python backend (`BertaBackend`, torch 2.14 on the CPU, the
weights from `wfzyx/von-1.0`); MiniCPM5-2B (`openbmb/MiniCPM5-2B-GGUF`,
Q8_0) through `llama-server`, one chat completion per case with the answer
constrained by a grammar to the option ids, thinking on or off through the
chat template.

| model | params | authored144 | balanced | perturbations108 | ms / case |
|---|---|---|---|---|---|
| **lev encoder** `english` | 395M | 61.1% (88) | 64.8% | 67.6% (73) | 117 (CPU) |
| **lev encoder** `typed-decisions` | 395M | 66.7% (96) | 68.1% | | 116 (CPU) |
| **lev encoder** `multilingual` | 307M | 59.0% (85) | 56.4% | | 48 (CPU) |
| von-1.0, NLI zero-shot | 395M | 76.4% (110) | 76.4% | 73.1% (79) | 160 (CPU, torch) |
| MiniCPM5-2B Q8 through llama-server, direct answer (grammar) | 2.5B | 73.6% (106) | 71.8% | | 566 (CPU) / 89 (Metal) |
| MiniCPM5-2B Q8 through llama-server, thinking, free-form answer | 2.5B | **97.2% (140)** | 97.6% | | 3,938 mean, 2,759 median (Metal); ~9,800 mean, 8,050 median (CPU, 24-case sample); ~300 tokens of thought |
| **lev thinker** (MiniCPM5-2B Q8 in the binary), thinking off | 2.5B | 74.3% (107) | 71.4% | | 146 (Metal) |
| **lev thinker**, thinking, sampled (temperature 1.0) | 2.5B | 91.0% (131) | 91.3% | | 5,388 mean, 3,926 median (Metal); 342 tokens |
| **lev thinker**, thinking, greedy (the default) | 2.5B | **95.1% (137)** | 95.5% | | 4,716 mean, 3,084 median (Metal); 369 tokens |
| [localjev](https://github.com/githubnext/localjev) + DiffusionGemma 26B-A4B 4-bit on oMLX 0.6.4 | 26B (4B active), 16.6 GB | 91.0% (131) | 90.0% | | 5,192 mean, 5,262 median (Metal) |
| lev thinker, Qwen3-4B-Thinking-2507 Q8 (unsloth GGUF), greedy, 4,096-token budget | 4B | 94.4% (136) | 95.3% | | 42,543 mean, 41,216 median (Metal); 1,883 tokens |

The lev thinker is the same model and the same llama.cpp inside
`lev-server` (`lev.think`): the prompt is built by lev, the thought is
decoded greedily until `</think>`, then `\n\nANSWER: ` is forced and each
option is scored by the log probability of its tokens (plus the end-of-turn
token), a softmax over those being the answer's probabilities — the same
calibrated shape the encoders give, where the llama-server harness parsed
a free-form final line. Greedy beats a sampled thought by four points here
and is reproducible; the two points to the free-form run are prompt and
decoding noise on seven cases.

localjev is a Bun bridge that turns the Jev request into a prompt asking
an MLX-served chat model for a JSON probability vector (no thinking,
temperature 0, retries on malformed JSON). Its probabilities are
self-reported: 143 of the 144 answers here were one-hot, and all 13 wrong
answers came with a probability of 0.9 or more, so its `confidence` says
nothing. It needs oMLX and bun on Apple Silicon and a 16.6 GB model; it
was slower than the 2.5B thinker in the lev binary and four points behind it.

## In-distribution: AG News, BoolQ, SST-5

The trio localjev's bake-off uses, 40 balanced cases each
(`bench/triad120.py` builds the set from the public datasets; `jolt -M
bench/triad.clj <model> [thinking]` runs it). Routing-style traffic, the
kind the encoder's presets exist for:

| model | AG News | BoolQ | SST-5 (acc / MAE) | all | ms per question |
|---|---|---|---|---|---|
| lev encoder `english` | **97.5%** | 72.5% | 27.5% / 1.08 | **65.8%** | **125** (CPU) |
| lev thinker, thinking off | 82.5% | 70.0% | 27.5% / 1.25 | 60.0% | 152 (Metal) |
| lev thinker, thinking (greedy) | 85.0% | **90.0%** | **37.5% / 0.80** | **70.8%** | 3,001 (Metal) |

Localjev's own numbers on the same three tasks (different samples, M5
Max): Gemma 4 26B-A4B 75.0% macro at 0.68 s, DiffusionGemma 26B-A4B
74.2% at 1.2 s, with a 2,048-word distraction dropping everything.

The encoder is the fast path: it beats the 2.5B model answering at once on
the routing task by 15 points, at 125 ms on a CPU against 150 ms on a GPU
(~560 ms on a CPU), one forward for a whole workflow instead of one prompt
per question, with calibrated confidence. The thinker's thinking earns
its seconds on the reading-comprehension and reasoning cases.

## Candidates from RESEARCH-2026-09.md, measured

- **Option-permutation averaging** (the note's #3; the encoder's answer
  averaged over every rotation of the option list): 20 of 144 cases change
  their argmax under rotation (14%, the same share nibzard measured on Jev).
  Averaging moves `english` 61.1% → 63.9% and its ECE 0.101 → 0.082;
  `typed-decisions` 66.7% → 66.0%. Three forwards per question, one batch.
  Small and inconsistent.
- **Neutral-state prior subtraction** (jevmlx's correction; score the
  options against an empty state, subtract the log-odds): `english` 61.1%
  → 54.9%, and combined with averaging 62.5%. An empty state is out of the
  encoder's distribution; the correction hurts. Not worth it.
- **Qwen3-4B-Thinking-2507 as the thinker** (#5): 94.4%, the same as
  MiniCPM5-2B's 95.1%, at 13× the time (41 s a case, 1,883 thought tokens
  against 369). The 2.5B model is the escalation tier.
- **The gate on in-distribution traffic** (#4): the encoder's noul
  confidence is `max(p, 1 − p)`, never below 0.5, and its BoolQ answers are
  over-confident — at a 0.5 threshold every BoolQ case stays on the
  encoder (72.5% right) where the thinker gets 90%; at 0.7 it still keeps
  37 of 40. Choice and score confidences (1 − normalised entropy) behave:
  at 0.5 the gate keeps 36/40 AG News (the encoder's best task) and 13/40
  SST-5 (its worst). The gate needs a per-type threshold or a refit noul
  temperature before it can be trusted on yes/no questions.
- **Calibration** on authored144: encoder ECE 0.101 (typed-decisions
  0.133), von 0.124, the thinker 0.039 — but 97% of the thinker's answers
  come with confidence above 0.5, so its confidence cannot gate anything;
  the encoder's can (88% of these adversarial cases sit below 0.5, 12% of
  in-distribution AG News).

## The fixes, measured (`lev.calibrate`, per-type thresholds, `debias`)

`jolt -M:calibrate --labels bench/data/calib.jsonl` (1,240 public cases
from `bench/calib_data.py`: SST-2, AG News, dair-ai/emotion, 20
newsgroups, BoolQ, SST-5 — one per temperature bucket), fitted on the even
cases, reported on the odd ones, `english`:

| bucket | n | accuracy | T shipped → refit | NLL before → after | ECE before → after |
|---|---|---|---|---|---|
| choice:2 (SST-2) | 87 | 95.4% | 1.91 → 1.24 | 0.169 → 0.159 | 0.060 → 0.033 |
| choice:3-5 (AG News) | 97 | 95.9% | 1.76 → 1.47 | 0.161 → 0.145 | 0.047 → 0.021 |
| choice:6-10 (emotion) | 127 | 39.4% | 1.00 → 3.00 | 2.929 → 1.539 | 0.468 → 0.172 |
| choice:11+ (20 newsgroups) | 104 | 39.4% | 0.10 → 1.48 | 11.02 → 2.06 | 0.567 → 0.103 |
| noul:2 (BoolQ) | 104 | 77.9% | 1.98 → 2.70 | 0.534 → 0.476 | 0.114 → 0.090 |
| score:3-5 (SST-5) | 101 | 35.6% | 1.25 → 4.02 | 2.002 → 1.458 | 0.290 → 0.101 |

The shipped temperatures are sharp where the encoder is weakest (a 20-way
choice at 35% accuracy reported ~1.0 confidence). On the trio the refit
takes ECE 0.161 → 0.106 and the gate at 0.5 now escalates every SST-5 case
(13/40 were kept before) while AG News keeps 38/40 at 97%. BoolQ does not
move: the noul confidence floor is 0.5 and within the task confidence
barely tracks correctness (kept accuracy 72% at ≥0.5, 79% at ≥0.9), so
yes/no questions need their own, higher threshold — `threshold`
{"noul": 0.9} — or, on traffic like this, escalating them outright (the
thinker gets 90%).

`--debias` through the runner: `english` 63.9%, ECE 0.082, 199 ms a case
(the rotations share one batched forward).

What the numbers say:

- The gap to a hosted generative decision API on hard cases is the
  thinking, not the encoder. Answering directly, the 2.5B decoder is no
  better than von (73.6% vs 76.4%) at 3.5x the CPU time; with ~300 tokens
  of reasoning it gets 140/144, at 3 s a case on the GPU and ~10 s on the
  CPU.
- von-1.0 is the same ModernBERT-large encoder this port runs, with an NLI
  head instead of the encoders' marker head, one pair sequence per option
  instead of one sequence per question. +15 points over the `english`
  encoder on this set for ~1.4x the time. The encoder under-predicts the abstaining option
  (`insufficient` 18 times against 36 expected); von's failures are
  concentrated in rule application under perturbation (10/36).
- The two encoders fail on different cases: 41 von-right/english-wrong, 19 the
  other way, 15 both wrong, and the thinker gets all 15 of those. A
  confidence gate over the encoder's top-2 margin at 0.5 sends 92 of 144 cases up
  to the thinker and lands at 88.9%; over von's at 0.5, 92 up and 93.1%.
  On this set almost everything is low-confidence; on routine traffic the
  gate would pass most cases through at encoder speed.

The Jev benchmark from the other session (24/64 for the encoder vs 60/64 hosted Jev)
is the same shape: a generative model with a reasoning budget against a
single-pass encoder.

## Multi-question calls: the state tokenized once, the prefixes cached

`bench/workflow.clj` times the README's four email questions as one call
(choice, score, noul, noul) on a short state (53 tokens) and a long one
(1,631 tokens, cut to max_len 512 for every question), 30–40 iterations
after a warmup, `english` on the M-series CPU. Before, `encoder-forward`
tokenized the state once per question; after (laya-mlx's PrefixCache
ported: `lev.sequence/encode-state` once a call, `cached-prefix` in a
128-entry LRU per agent), once per call, and a question asked before
costs no tokenization at all. Answers are unchanged (golden/ pins them:
`prefix-and-state-assemble-into-build-sequence` proves the ids byte for
byte).

| call | before, p50 (two runs) | after, p50 (two runs) |
|---|---|---|
| tokenize the 1.6k-token state, alone | 57 ms | 58 ms |
| 4 questions, short state | 288 / 304 ms | 296 / 302 ms |
| 4 questions, long state | 1,392 / 1,381 ms | 1,247 / 1,239 ms |
| 1 question, long state | 400 / 374 ms | 391 / 398 ms |

The long-state call drops 10%: three of its four 48 ms state
tokenizations were redundant. The short-state case saves ~5 ms of a 300
ms call, under the run-to-run drift of the same code (288 vs 304 ms), so
no claim there. The tokenizer itself runs at ~33 µs a token, slow in
absolute terms (laya-jolt-edf); this removes the repetition, not the rate.
