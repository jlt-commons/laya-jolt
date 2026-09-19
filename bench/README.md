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
jolt -M bench/authored144.clj --file bench/data/perturbations108.jsonl --out results.jsonl
```

Each row is one `choice` question: the row's question as the instructions,
its options as the criteria, its state as the state.

## Results, 2026-09-19

10-core M-series laptop. laya-jolt through `bench/authored144.clj`; von-1.0
through its own Python backend (`BertaBackend`, torch 2.14 on the CPU, the
weights from `wfzyx/von-1.0`); MiniCPM5-2B (`openbmb/MiniCPM5-2B-GGUF`,
Q8_0) through `llama-server`, one chat completion per case with the answer
constrained by a grammar to the option ids, thinking on or off through the
chat template.

| model | params | authored144 | balanced | perturbations108 | ms / case |
|---|---|---|---|---|---|
| laya `english` (this port) | 395M | 61.1% (88) | 64.8% | 67.6% (73) | 117 (CPU) |
| laya `typed-decisions` (this port) | 395M | 66.7% (96) | 68.1% | | 116 (CPU) |
| laya `multilingual` (this port) | 307M | 59.0% (85) | 56.4% | | 48 (CPU) |
| von-1.0, NLI zero-shot | 395M | 76.4% (110) | 76.4% | 73.1% (79) | 160 (CPU, torch) |
| MiniCPM5-2B Q8, direct answer | 2.5B | 73.6% (106) | 71.8% | | 566 (CPU) / 89 (Metal) |
| MiniCPM5-2B Q8, thinking | 2.5B | **97.2% (140)** | 97.6% | | 3,938 mean, 2,759 median (Metal); ~9,800 mean, 8,050 median (CPU, 24-case sample); ~300 tokens of thought |

What the numbers say:

- The gap to a hosted generative decision API on hard cases is the
  thinking, not the encoder. Answering directly, the 2.5B decoder is no
  better than von (73.6% vs 76.4%) at 3.5x the CPU time; with ~300 tokens
  of reasoning it gets 140/144, at 3 s a case on the GPU and ~10 s on the
  CPU.
- von-1.0 is the same ModernBERT-large encoder this port runs, with an NLI
  head instead of laya's marker head, one pair sequence per option instead
  of one sequence per question. +15 points over laya `english` on this set
  for ~1.4x the time. laya under-predicts the abstaining option
  (`insufficient` 18 times against 36 expected); von's failures are
  concentrated in rule application under perturbation (10/36).
- The two encoders fail on different cases: 41 von-right/laya-wrong, 19 the
  other way, 15 both wrong, and the thinker gets all 15 of those. A
  confidence gate over laya's top-2 margin at 0.5 sends 92 of 144 cases up
  to the thinker and lands at 88.9%; over von's at 0.5, 92 up and 93.1%.
  On this set almost everything is low-confidence; on routine traffic the
  gate would pass most cases through at encoder speed.

The Jev benchmark from the other session (24/64 laya vs 60/64 hosted Jev)
is the same shape: a generative model with a reasoning budget against a
single-pass encoder.
