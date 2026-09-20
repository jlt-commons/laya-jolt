# Open models comparable to Jev — survey and what it means for lev

September 2026. Sources: TypeSafe's blog and evals, the laya model card, four
open Jev-compatible implementations, two independent Jev benchmarks, the
BTZSC zero-shot classification study (ICLR 2026), GLiClass (arXiv 2508.07662),
the localjev bake-off, and one independent 16-model measurement (WotAI).

## What Jev actually delivers, per independent measurement

TypeSafe's own evals (evals.typesafe.ai) score Jev against the *consensus of
GPT-6 Astra + Claude Fable 5.1* on four workflows — agreement with frontier
LLMs, not correctness. Independent measurements:

- **nibzard/decision-model-benchmark** (5 suites, raw logs published): jev
  76.3% on banking77-style 77-way intent vs gpt-oss-120b 81.3% and glm-5.3
  80.4%; p50 264–276 ms; cheapest ($0.07/1k); hard 255-option cap; admits
  ignorance on only 49.7% of forced-uncertainty items (LLMs: 97–100%), ECE
  0.246 — the worst measured; 13% of choices flip under option permutation.
- **AbdelStark/jev-benchmarks** (300 BTZSC examples): jev AG News 0.910,
  Banking77 0.870, DAIR Emotion 0.480; zero probability on the true label on
  16% of Emotion examples.
- **WotAI** (150 passages, 16 models): Jev 66.0% accuracy, ECE 0.121, p50
  455 ms; ties Claude Haiku 4.5 on accuracy and ECE; beats every other
  sub-second model on calibration and is the only sub-second model that
  flags uncertainty often (34.7% of rows "unsure").

Net: Jev is real on speed/cost and honest-ish uncertainty, mid-pack on raw
accuracy, and weak on adversarial abstention — exactly the shape of lev's
bench/ findings (the gap to generative models is the reasoning budget, not
the encoder).

## Open models that perform comparably

Same wire API (`POST /v1/systemone`, TypeSafe SDK works unchanged):

| project | model | approach | measured vs Jev |
|---|---|---|---|
| **OpenJev** (razorback16/openjev, Apache-2.0) | DiffusionGemma 26B-A4B (NVFP4, vLLM PR #57250) | answer template written onto the diffusion canvas; one read-only denoise step reads full distributions off the label slots; re-reads (up to 4) and averages when entropy > 0.1; extensions: `images`, `steps`, `samples`, `think` 0–4096, `sequential` | served free on Codiv; 38 ms typical; no published accuracy vs Jev yet |
| **kev** (jaredpalmer/kev) | Qwen3-0.5B/0.6B/4B/8B + LoRA + pointer head | state and all questions packed in one sequence, block-causal mask (questions can't see each other), pointer head scores options off each question's decision token; cross-entropy trained probabilities | out-of-domain locked test: kev-4b 0.852, kev-8b 0.869, kev-8b Brier 0.34 OOD, vs real Jev 0.86 on same items; serves on a 32 GB Mac |
| **laya** (convaiinnovations/laya, Apache-2.0) | ModernBERT-large / mmBERT-base — *lev's own checkpoints* | single forward, marker head, RLCD-trained calibration | card claims (3rd-party Jev numbers): AG News 0.950 vs 0.910, Emotion 0.595 vs 0.480, ECE 0.081 vs 0.246, p50 32.8 ms vs 236–276 ms; typed-decisions 0.766 vs 0.727 — but that is the *fine-tuned* checkpoint; zero-shot on that suite is 0.362, below majority class |
| **jevmlx** (bnsd55/jevmlx, MIT) | Qwen2.5-1.5B/3B/7B-Instruct 4-bit (MLX) | schema of booleans/enums/multi-selects scored in one batched forward pass; prior correction (neutral-context prior subtraction), fitted calibrators, constrained-MAP reconciliation, ordinal telemetry, abstain-by-margin, NONE_OF_ABOVE | no Jev comparison published; engineering is the interesting part |
| **localjev** (githubnext/localjev, MIT) | any oMLX chat model (default DiffusionGemma 26B-A4B) | bridge: prompt for a JSON probability vector, validate, retry | authored144 (bench/README): 91.0% with a 26B-A4B 4-bit, slower than lev's 2.5B thinker; probabilities self-reported (143/144 one-hot) — calibration meaningless |

Not wire-compatible but direct competitors as classifiers:

| family | best open entries | evidence |
|---|---|---|
| **GLiClass** (GLiNER adapted to classification) | gliclass-large-v3.0 (DeBERTa-v3-large, 0.72 avg F1), modern-base/edge variants; single forward pass, all labels at once, throughput drops only 7–20% from 1→128 labels where cross-encoders drop ~50×; NONE_OF_ABOVE and 8-shot support built in | GLiClass paper; +5.5% over best zeroshot-v2.0 cross-encoder |
| **NLI cross-encoders** | MoritzLaurer/ModernBERT-large-zeroshot-v2.0, tasksource/ModernBERT-large-nli, deberta-v3-large-nli-triplet | BTZSC: NLI plateaus with backbone size; 0.55–0.59 avg F1; von-1.0 (bench/) is this family at 76.4% on authored144 |
| **Rerankers** | **Qwen3-Reranker-0.6B** (beats all NLI cross-encoders) and **Qwen3-Reranker-8B** (0.72 macro F1, SOTA zero-shot) | BTZSC/ICLR 2026; 0.6B is the size-efficient pick |
| **Embedding zero-shot** | GTE-large-en-v1.5 | closes most of the NLI gap, best accuracy/latency trade-off, but no per-label text interaction |
| **Small thinking decoders** | MiniCPM5-2B (lev's thinker, 95.1% authored144 greedy), Qwen3-4B-Thinking-2507 (AIME25 81.3, MMLU-Pro 74 vs MiniCPM5-class), Qwen3-30B-A3B-Thinking | authored144: thinking closes the abstention/negation gap (61–76% → 95–97%) |

The localjev bake-off (same AG News / BoolQ / SST-5 trio as lev's 120-case
set, 40 examples/task, JSON-probability prompting): best macro accuracy was
Qwen3.6-35B-A3B **76.7%** and Gemma 4 26B-A4B **75.0%**; lev encoder
`english` macro is **65.8%** (97.5 / 72.5 / 27.5). SST-5 stays the weak
primitive everywhere (best model: 55%). With 2,048 distraction words every
model degrades (up to −37.5 points on SST-5) — length discipline matters.

## What this means for lev — improvement candidates

Ranked by expected gain over effort, grounded in the sources above.

1. **Per-(type, option-count) calibration refit as a first-class tool.**
   The laya card: shipped checkpoints are over-confident; refitting one
   temperature per (question type, option count) moves mean ECE 0.466 →
   0.081. lev already fits calibration constants at prepare time from the
   checkpoint; a `lev.calibrate` that refits on a user's labeled data
   (jevmlx ships exactly this: `jevmlx calibrate --out`, calibrator bundle
   selectable at decision time) would make the confidence gate trustworthy
   on real traffic. nibzard's jev ECE 0.246 vs laya's refit 0.081 is the
   size of the prize.

2. **Abstention / "insufficient evidence" handling.** bench/ already shows
   the encoder under-predicts the abstaining option (18 vs 36 expected on
   authored144); nibzard shows the same failure in Jev itself (admits
   ignorance 49.7% where LLMs do 97–100%). Concrete: jevmlx's
   `abstain_below_margin` (withhold the value, keep the raw one alongside)
   and GLiClass's explicit NONE_OF_ABOVE option as an auto-addable option.
   lev could offer both: an optional abstain threshold per question, and an
   `:none-of-above true` question flag that adds the option and maps it to
   `null`.

3. **Option-order debiasing via prior correction.** nibzard measured 13%
   choice flips under option permutation on Jev; jevmlx subtracts a
   neutral-context prior (score the options against an empty/neutral state
   once, subtract those log-odds) and OpenJev re-reads with fresh noise and
   averages when entropy > 0.1. The encoder analogue is cheap: one extra
   batched forward with the same questions over a neutral state, subtract
   per-option priors from the logits before the calibrated softmax; or
   average calibrated probabilities over 2–4 option permutations (2–4×
   cost, still ~100 ms/class). This attacks position bias and the
   under-predicted abstention option at once.

4. **A `confidence` that is defined and measured.** OpenJev defines
   confidence as `1 − H(p)/ln K`; WotAI's whole ranking hinges on an
   "unsure" share; kev reports Brier and confident-error rate as headline
   numbers. lev's docs describe confidence qualitatively; bench/ should
   report ECE + top-2 margin histograms next to accuracy so gate thresholds
   are chosen against data (the 0.5-margin gate sending 92/144 up is a
   calibration symptom, per #1/#3).

5. **Stronger thinker candidate: Qwen3-4B-Thinking-2507.** Q4_K_M ≈ 2.5 GB,
   Apache-2.0, a real thinking mode, and materially stronger reasoning than
   the 2.5B class (MMLU-Pro 74.0, AIME25 81.3). lev's thinker interface
   already takes any GGUF; what is needed is a ChatML `<think>` template
   entry in `lev.think/defaults` and an authored144 run to compare against
   MiniCPM5-2B's 95.1%. kev-8b's 0.869 OOD suggests 4B-class decoders are
   where the accuracy/cost frontier sits for the escalation tier.

6. **GLiClass-style encoder as a third model kind.** lev already ports
   GLiNER2's constrained classification (`lev.constraints`); GLiClass is
   the same family trained for classification, single forward pass with all
   labels, 0.72 avg F1, NONE_OF_ABOVE and few-shot built in. A
   `gliclass-large-v3.0` served behind the same `system-one` API would
   cover label-heavy tasks (laya warns accuracy falls past ~20 options;
   GLiClass degrades only 7–20% out to 128 labels). Alternatively
   Qwen3-Reranker-0.6B as a scorer for high-cardinality choices (BTZSC:
   beats all NLI cross-encoders at 0.6B).

7. **Sequential question chunks with earlier answers visible.** OpenJev's
   `sequential` extension: for long question lists, read chunks in order,
   each seeing the answers already chosen. Complements lev's joint
   constraints decoder: constraints decide *after* the pass today; a
   sequential mode would let question 20 condition on question 1's decided
   answer. Cheaper than thinking, closes part of the multi-step gap.

8. **Long-input discipline, measured.** localjev: every model degraded with
   2k distraction words (SST-5 −37.5 points worst case); lev's state
   truncation is silent from the end. `usage.input_tokens` exists; a
   one-line `truncated: true` / `state_tokens_dropped: n` on the answer
   when the state is cut would surface it per request instead of per
   investigation.

## Non-improvements (checked, not worth it)

- **DiffusionGemma structured read (OpenJev's path)**: needs vLLM PR #57250
  and an NVIDIA 24 GB+ GPU; lev's CPU/laptop positioning and llama.cpp
  native don't fit it. Watch it — if the PR merges and small diffusion
  models land, it's the architecture to reassess.
- **JSON-probability prompting (localjev's path)**: self-reported
  probabilities, one-hot 143/144, four points behind lev's thinker on
  authored144 at higher cost. Already beaten.
- **Scaling the NLI/von encoder family**: BTZSC shows NLI cross-encoders
  plateau with backbone size; the win comes from reranker/classifier-style
  training (see #6) or thinking (#5).
- **Larger localjev-style MoE (26B-A4B)**: 16.6 GB, slower than lev's
  2.5B thinker, no real probabilities. The thinker tier is better served by
  a 4B thinking model.
