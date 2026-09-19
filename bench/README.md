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
