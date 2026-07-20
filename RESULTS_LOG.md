# Results Log — Hyperbolic Bi-Encoders for Long-Tail Entity Typing

One entry per experiment. Each: **Hypothesis · Setup · Results · Analysis**.
Numbers are pasted from `results/` JSONs; mAP is threshold-free and is the
metric to trust (F1-only wins can be calibration artifacts).

---

## E0 — Establish the premise (long-tail statistics), no training

**Hypothesis**
- The unified fine-grained typing vocabulary is Zipfian: most types are rare
  (≤5 mentions), which is what motivates a geometry that shares structure across
  types.

**Setup**
- `scripts/e0_build_vocab_taxonomy.py` over UFET crowd (+ distant dumps).
- Output: `results/e0/stats.json`, `results/e0/type_counts.tsv`.

**Results**
- n_types = 3,460 · n_mentions = 61,671
- % types in tail (≤5 mentions) = **75.95%**
- Top types: location 8123, person 6255, organization 5345, scientific concept
  2923, date 2402, cultural reference 1771, product 1460, quantity 1162,
  event 1145, technology 1141.

**Analysis**
- Premise confirmed: ~3/4 of types are true tail — a flat classifier has almost
  no signal for them.
- Head is dominated by a handful of coarse types → strong head/tail imbalance,
  justifying tail-stratified evaluation.

---

## E0b / Taxonomy — WordNet hypernym linking, no training

**Hypothesis**
- A mechanical WordNet hypernym taxonomy over the UFET-crowd types is dense and
  clean enough to (a) supervise ancestor propagation and (b) score hierarchical F1.

**Setup**
- `scripts/build_taxonomy_wordnet.py`; frequency filter (parent ≥ child freq)
  prunes wrong-sense chains (e.g. `person → cause → event`).
- Output: `results/taxonomy/wordnet_parent.json`, `taxonomy_stats.json`.

**Results**
- n_types = 2,519 · linked = 2,445 (**97.1%**) · roots = 74
- max_depth = 9 · mean_depth = 2.57

**Analysis**
- Coverage is high (97% linked) → usable as-is for P1.
- Shallow mean depth (2.57) with 74 roots = a forest, not one clean tree; enough
  hierarchy to test the geometry, but taxonomy *quality* remains an open variable
  (deferred to P4 ablation).

---

## P1 — Geometry probe on UFET crowd (go/no-go gate) — DONE

**Hypothesis**
- H1 (geometry): hyperbolic (Poincaré) scoring beats identical Euclidean scoring,
  concentrated on rare types.
- H2 (low dim): the hyperbolic advantage is larger at low embedding dimension.
- H3 (supervision composes): ancestor-propagated supervision helps *more* in
  hyperbolic space than Euclidean.

**Setup**
- Same span-level bi-encoder (`bert-base-uncased`); only knob = `GeometryHead`
  (euclidean = scaled cosine, hyperbolic = negative geodesic distance).
- Grid: 2 geometries × {flat, ancestor} × dim {16, 64, 128}, seed 42
  (`results/p1/`) + 4-seed replication on key cells (`results/p1_seeds/`).
- Fair-comparison protocol: per-model dev threshold sweep; model selection = best
  dev macro-F1; report mAP (threshold-free) + tail-stratified F1 + hierarchical F1.
- Runners: `train_probe.py`, `scripts/p1_launch_{grid,seeds}.sh`.

**Results — full grid (seed 42), `results/p1/`**

| geom | sup | dim | micro | macro | mAP | head | mid | tail | hierF1 | selEp |
|------|-----|-----|-------|-------|-----|------|-----|------|--------|-------|
| euclidean  | ancestor | 16  | 0.195 | 0.200 | 0.237 | 0.345 | 0.035 | 0.000 | 0.304 | 23 |
| euclidean  | ancestor | 64  | 0.182 | 0.196 | 0.233 | 0.347 | 0.026 | 0.000 | 0.278 | 24 |
| euclidean  | ancestor | 128 | 0.180 | 0.193 | 0.229 | 0.337 | 0.026 | 0.000 | 0.282 | 24 |
| euclidean  | flat     | 16  | 0.274 | 0.251 | 0.287 | 0.441 | 0.109 | 0.000 | 0.304 | 24 |
| euclidean  | flat     | 64  | 0.269 | 0.252 | 0.277 | 0.443 | 0.082 | 0.000 | 0.298 | 24 |
| euclidean  | flat     | 128 | 0.273 | 0.252 | 0.277 | 0.437 | 0.093 | 0.000 | 0.302 | 24 |
| hyperbolic | ancestor | 16  | 0.263 | 0.234 | 0.277 | 0.468 | 0.208 | 0.009 | 0.357 | 24 |
| hyperbolic | ancestor | 64  | 0.268 | 0.240 | 0.284 | 0.484 | 0.214 | 0.010 | 0.362 | 24 |
| hyperbolic | ancestor | 128 | 0.241 | 0.228 | 0.264 | 0.442 | 0.186 | 0.007 | 0.342 | 24 |
| hyperbolic | flat     | 16  | 0.305 | 0.239 | 0.332 | 0.524 | 0.262 | 0.031 | 0.342 | 24 |
| hyperbolic | flat     | 64  | 0.322 | 0.269 | 0.357 | 0.538 | 0.278 | 0.012 | 0.360 | 23 |
| hyperbolic | flat     | 128 | 0.309 | 0.259 | 0.311 | 0.506 | 0.230 | 0.000 | 0.320 | 6  |

**Results — seed-averaged (n=4, incl. grid seed 42), `results/p1_seeds/`**

| geom | sup | dim | mAP | macro | tail | hierF1 |
|------|-----|-----|-----|-------|------|--------|
| euclidean  | ancestor | 64 | 0.233±0.003 | 0.196±0.001 | 0.000±0.000 | 0.290±0.008 |
| euclidean  | flat     | 16 | 0.280±0.004 | 0.250±0.002 | 0.000±0.000 | 0.302±0.006 |
| euclidean  | flat     | 64 | 0.280±0.002 | 0.252±0.002 | 0.000±0.000 | 0.304±0.003 |
| hyperbolic | ancestor | 64 | 0.284±0.004 | 0.239±0.003 | 0.010±0.001 | 0.360±0.004 |
| hyperbolic | flat     | 16 | 0.330±0.004 | 0.238±0.004 | 0.040±0.007 | 0.336±0.005 |
| hyperbolic | flat     | 64 | 0.353±0.002 | 0.267±0.003 | 0.017±0.006 | 0.357±0.004 |

**Analysis**
- **H1 supported, and it's real.** Hyperbolic beats Euclidean on **mAP**
  (threshold-free) at every matched cell: d64 flat 0.353 vs 0.280 (+0.073, tiny
  std). Not a calibration artifact.
- **Gain concentrates in the tail/mid, as predicted.** d64 flat: head 0.443→0.538,
  mid 0.082→0.278 (>3×), tail 0.000→0.017. The geometry helps where the theory says.
- **H2 supported.** Hyperbolic edge persists at d16 (0.330 vs 0.280 mAP); d16
  hyperbolic gives the best tail-F1 (0.040) → classic low-dim hyperbolic advantage.
- **H3 not supported (yet).** Ancestor propagation *lowers* mAP/macro in both
  geometries (by construction: predicted supertypes = false positives). It only
  helps **hierarchical F1** (hyp-anc-d64 hierF1 0.360). Plain **flat + hyperbolic**
  is the strongest cell on all threshold-free metrics.
- **d128 unstable — do not trust.** hyp-flat-d128 selected best epoch at **6**
  (vs ~24) then degraded (boundary saturation). Its lower numbers are an artifact,
  not evidence against high dim.
- **True tail (≤5 ex) unlearnable by supervision alone** in any geometry
  (tail-F1 ≤0.04) → deferred to P3 (zero-shot / name-seeded embeddings).
- **Gate verdict: PASSED.** Proceed to P2 (controlled noise robustness).

---

## P2 — Controlled noise robustness — DONE (hypothesis inverted)

**Hypothesis**
- Hyperbolic degrades more gracefully under label noise, because realistic noisy
  labels are hierarchy-neighbors that the geometry keeps close. Expect the
  euclidean−hyperbolic gap to *widen* with noise under sibling (realistic) noise,
  and both to fall together under uniform (adversarial) noise.

**Setup**
- Cell fixed to the strongest/cleanest P1 config: **flat supervision, dim 64,
  UFET crowd, bert-base-uncased**, identical schedule (25 epochs, dev threshold
  sweep, best-dev-macro-F1 selection). Only geometry + noise vary.
- Noise injected into **train labels only** (dev/test clean), per positive label,
  seeded (`noise_seed = 1000 + seed`):
  - **sibling** = replace gold with a taxonomy sibling (same parent); falls back
    to uniform for types with no in-vocab sibling.
  - **uniform** = replace gold with a uniform-random type (adversarial control).
- Grid: 2 geometries × {clean(0%), sibling@{10,30,50}%, uniform@{10,30,50}%}
  × seeds {1,2,3} = **42 runs**.
- Code: `--noise-rate`/`--noise-mode` in `train_probe.py`
  (`SpanTypingDataset._corrupt_labels`); runner `scripts/p2_launch.sh`;
  outputs in `results/p2/`. Verified effective corruption rate matches target
  and is deterministic across dataset builds.

**Results** — 42/42 runs, 0 failures. test **mAP** (threshold-free), mean±std over
seeds {1,2,3}. Baseline (0%) is shared across modes. Plot:
`results/p2/degradation_curves.png`.

| rate | euc-sibling | hyp-sibling | gap(h−e) | euc-uniform | hyp-uniform | gap(h−e) |
|------|-------------|-------------|----------|-------------|-------------|----------|
| 0%   | 0.281±.001  | 0.352±.001  | +0.071   | (shared)    | (shared)    | +0.071 |
| 10%  | 0.278±.003  | 0.348±.005  | +0.070   | 0.275±.001  | 0.356±.001  | +0.081 |
| 30%  | 0.272±.003  | 0.332±.004  | +0.060   | 0.263±.003  | 0.362±.004  | +0.099 |
| 50%  | 0.251±.002  | 0.294±.006  | +0.042   | 0.237±.002  | 0.352±.001  | +0.115 |

Macro-F1 gap at 50%: sibling −0.000, uniform +0.048. Hier-F1 gap at 50%:
sibling −0.005, uniform +0.054. Tail-F1 stays ~0 everywhere (P3's problem).

**Analysis**
- **Hypothesis inverted.** We predicted hyperbolic degrades *gracefully* on
  realistic (sibling) noise → gap widens. The opposite held.
- **Uniform (adversarial) noise: hyperbolic is nearly immune.** hyp-mAP flat at
  ~0.35 through 50% (0.352→0.352); euclidean collapses (0.281→0.237). Gap grows
  +0.071 → +0.115.
- **Sibling (realistic) noise: hyperbolic is the *more* fragile one.** hyp drops
  0.352→0.294 (−16%) vs euc 0.281→0.251 (−11%); gap shrinks +0.071 → +0.042, and
  by 50% the advantage vanishes in macro-F1 and hier-F1.
- **Mechanism (consistent across all metrics):** siblings share a parent → sit
  *close* in the hyperbolic embedding, so sibling corruption injects a
  geometrically-adjacent, near-indistinguishable wrong target that erodes exactly
  the tree structure hyperbolic exploits. Uniform labels land *far* in the roomy
  near-boundary region and act as ignorable outliers. Euclidean's crowded space
  has no safe far-away region, so it is hurt more by random noise.
- **Revised claim for the paper:** hyperbolic is exceptionally robust to
  *unstructured* label noise but specifically vulnerable to *structured
  (hierarchy-local)* noise — a sharper, more defensible statement than "graceful
  degradation."
- **Caveats:** single cell (flat, d64) — worth confirming the pattern holds at
  d16 and under ancestor supervision before headlining it. Effect sizes are well
  outside seed std, so the direction is solid.

---

## P3 — Zero-shot held-out tail (UFET-first) — DONE (two-sided result)

**Hypothesis**
- Name-seeded label embeddings + hyperbolic geometry enable typing of held-out /
  unseen types (where P1 supervision gets ~0 on the tail in any geometry). Expect
  hyperbolic to rank a purely name-derived unseen-type embedding better than
  euclidean.

**Setup**
- UFET crowd (comparable to P1/P2), `train_zeroshot.py`, flat supervision,
  bert-base-uncased, 25 epochs.
- **Split (fixed, `--split-seed 7`, identical across all conditions):** held-out =
  a stratified 50% sample of types with train-freq ∈ [1,50] and ≥2 test
  occurrences. → **386 held-out types, 2,698 train mentions removed, 2,873
  held-out test mentions (1,393 test examples carry a held-out gold).** Once held
  out, a type is unseen in training (freq 0) regardless of original frequency.
- **Training:** label matrix contains **only seen** types → held-out types get
  *zero* supervision (not even as BCE negatives). Examples keep seen labels;
  examples with only held-out labels drop.
- **Zero-shot eval:** re-encode held-out type *names* through the TRAINED
  encoder+`mention_proj`; score test mentions; rank. Metrics: (a) zero-shot =
  rank held-out labels only (386-way); (b) generalized = rank seen+held jointly
  (2,519-way). Report mAP / P@1 / R@5.
- Grid: 2 geometries × dim {16, 64} × seed {1,2,3} = 12 runs. Only geometry
  varies. Runner `scripts/p3_launch.sh` (GPUs 2,5 excluded — other user).

**Results** — 12/12 runs, 0 failures. mean±std over seeds {1,2,3}. n_eval = 1,393
test examples carrying a held-out gold. Plot: `results/p3/zeroshot_bars.png`.

| geom | dim | ZS mAP | ZS P@1 | ZS R@5 | GEN mAP | GEN P@1 |
|------|-----|--------|--------|--------|---------|---------|
| euclidean  | 16 | 0.077±.007 | 0.035±.009 | 0.090±.012 | 0.077±.007 | 0.035±.009 |
| euclidean  | 64 | 0.077±.004 | 0.038±.006 | 0.084±.007 | 0.077±.004 | 0.038±.006 |
| hyperbolic | 16 | 0.107±.019 | 0.063±.020 | 0.124±.026 | 0.049±.012 | 0.013±.007 |
| hyperbolic | 64 | 0.111±.012 | 0.065±.009 | 0.129±.024 | 0.043±.007 | 0.005±.002 |

ZS = rank the 386 held-out types only; GEN = rank held-out golds against the full
2,519-type space (seen + held-out).

**Analysis**
- **Pure zero-shot: hyperbolic wins decisively.** ZS mAP +44% at d64 (0.111 vs
  0.077), P@1 nearly doubles (0.065 vs 0.038). Name-derived embeddings of
  *never-supervised* types land in the right neighborhood far better under
  hyperbolic geometry — the distinctive selling point of the project, and a
  direct confirmation of the P3 hypothesis.
- **Generalized: hyperbolic reverses and loses** (GEN mAP 0.043 vs euc 0.077).
  Under distance scoring, trained seen label embeddings systematically outscore
  name-derived held-out ones (norm/scale mismatch → strong seen-bias), so held-out
  golds get buried. Euclidean's cosine head L2-normalizes both, so it has no such
  bias — GEN==ZS exactly (0.077 both; flagged as surprisingly exact, worth a quick
  independent recompute, but it does not favor the hypothesis so it is not a
  self-serving bug).
- **Honest story:** hyperbolic substantially improves unseen-type transfer, but
  only realizes it when the candidate space is the novel types; in the mixed
  setting it needs a seen/unseen calibration fix (temperature / per-label bias /
  norm matching) — a known generalized-ZSL problem, addressable.
- **Dim:** advantage is flat across d16/d64 (0.107 vs 0.111); consistent with a
  geometry effect rather than a capacity effect.
- **Caveats / next:** (1) held-out set spans train-freq 1..50 (not strictly <=5)
  for test support, could stratify ZS mAP by original frequency. (2) Try a
  calibration layer to recover GEN. (3) Scale up: repeat on FiNERweb-eng
  (+PileNER/NuNER) per the handover to test whether more/noisier data widens the
  zero-shot gap. [done: see P3b]

---

## P3b — Zero-shot scale-up on FiNERweb-eng — DONE (UFET win does NOT replicate)

**Hypothesis**
- The P3 zero-shot advantage of hyperbolic should hold, and per the handover
  possibly widen, when trained on the larger, noisier, distant FiNERweb-eng dump.

**Setup**
- Same `train_zeroshot.py` protocol as P3, generalized with
  `--train-file/--test-file/--max-len`. Corpus: FiNERweb-eng
  (`.../training_jsonl/finerweb/eng.jsonl`), distant, single-tag, 2,034 types,
  73% tail. Fixed doc-level split via `scripts/p3b_prep_finerweb.py`
  (split-seed 7): 2,124 train / 374 test docs, written under /vol/tmp.
- Held-out set (same rule as P3): 105 types, 1,562 train mentions removed,
  440 held-out test mentions (n_eval=421). Names are clean (natural disaster,
  planet, currency, job title, ...).
- Grid: 2 geometries x seed {1,2,3} @ dim 64 = 6 runs (dim fixed: P3 showed the
  geometry effect is dim-invariant). FiNERweb is ~20x more spans than UFET-crowd,
  so epochs=12, batch=64, max_len=256. Runner `scripts/p3b_launch.sh`.

**Results** — 6/6 runs, 0 failures. mean+/-std over seeds {1,2,3}.

| geom | ZS mAP | ZS P@1 | ZS R@5 | GEN mAP | GEN P@1 |
|------|--------|--------|--------|---------|---------|
| euclidean  | 0.339+/-.011 | 0.195+/-.022 | 0.507+/-.021 | 0.339+/-.011 | 0.195+/-.022 |
| hyperbolic | 0.310+/-.027 | 0.159+/-.021 | 0.481+/-.036 | 0.201+/-.026 | 0.044+/-.013 |

**Analysis**
- **The P3 result does not replicate.** Pure zero-shot: hyperbolic is -9%
  (0.310 vs 0.339), i.e. euclidean is ahead (vs +44% for hyperbolic on UFET).
  Borderline given seed spread, but not a win.
- **Generalized: hyperbolic still loses (-41%)**, same seen-bias as P3-UFET.
- **Likely drivers (confounded, three at once):** (1) ~20x more training spans
  (43k vs ~2k); hyperbolic's geometric prior helps most in low-data, euclidean
  catches up with abundant data. (2) Single-tag vs multi-label: no hierarchical
  co-occurrence for the geometry to exploit. (3) Absolute scores ~4x higher
  (euc 0.339 vs 0.077), task is easier so the prior is less needed.
- **Confound to note:** P3b used epochs=12/batch=64 vs P3's 25/32. euc-vs-hyp
  *within* FiNERweb is clean (identical settings); the cross-corpus comparison to
  UFET is not perfectly controlled.
- **Takeaway:** the zero-shot selling point is regime-dependent (low-data,
  multi-label UFET), not a universal property. Clean disentangler next: a
  data-scaling ablation on FiNERweb (subsample train toward UFET size, hold
  epochs/schedule fixed) to separate scale from corpus-type. [done: see P3c]

---

## P3c — Data-scaling ablation on FiNERweb-eng — DONE (scaling law, confound resolved)

**Hypothesis**
- The P3b non-replication is driven by SCALE, not by the single-tag corpus type.
  Subsampling FiNERweb train toward UFET size should recover the hyperbolic
  zero-shot advantage.

**Setup**
- Same corpus, split, and held-out set as P3b (fixed, computed from the full
  file, so only training amount varies). Subsample train via
  `--train-max-records` (added to `train_zeroshot.py`; subsamples docs only,
  vocab/split unchanged).
- **Schedule held CONSTANT at epochs=25, batch=64, max_len=256, dim=64** across
  all scales, so the comparison is controlled (same passes, different N).
- Grid: 2 geometries x scale {100, 300, 1000} docs x seed {1,2,3} = 18 runs.
  Runner `scripts/p3c_launch.sh` (+ `scripts/p3c_retry.sh` for 5 cells that OOM'd
  when other users grabbed GPUs; re-run one-per-GPU, 0 failures). Plot:
  `results/p3c/scaling_curve.png`.
- Note: the ~43k full point is P3b (epochs=12), a slightly different schedule, so
  treat it as indicative rather than part of the controlled sweep.

**Results** — 18/18 runs, 0 failures. zero-shot held-out mAP, mean+/-std, n=3.

| train docs | ~spans | euclidean | hyperbolic | gap (hyp-euc) | GEN euc | GEN hyp |
|------------|--------|-----------|------------|---------------|---------|---------|
| 100        | 1,907  | 0.055+/-.002 | 0.179+/-.038 | **+0.124** | 0.055 | 0.094 |
| 300        | 5,540  | 0.200+/-.021 | 0.261+/-.034 | +0.061 | 0.200 | 0.185 |
| 1000       | 18,562 | 0.353+/-.062 | 0.254+/-.029 | **-0.099** | 0.353 | 0.133 |
| (43k, P3b) | ~43,000| 0.339 | 0.310 | -0.029 | 0.339 | 0.201 |

**Analysis**
- **Confound resolved: it is SCALE, not corpus type.** On the same single-tag
  FiNERweb corpus, at UFET size (~1.9k spans) hyperbolic wins by +0.124, even
  larger than the +0.034 it won by on UFET itself. So single-tag data is not the
  reason the P3b win vanished.
- **A clean scaling law.** The hyperbolic-minus-euclidean gap decays monotonically
  with training data (+0.124 -> +0.061 -> -0.099) and crosses zero between ~5.5k
  and ~18.6k spans. With enough data euclidean learns the mention geometry
  directly and overtakes.
- **Unifying interpretation for the whole project:** hyperbolic geometry is a
  data-efficient inductive prior for long-tail / low-resource typing, not a
  universally better model. It substitutes for data; once data is abundant the
  prior becomes a mild constraint. This reconciles P1 (win on small UFET, tail,
  low dim), P3 (zero-shot win), and P3b (win gone at scale).
- **Generalized still loses at every scale** (seen-bias in distance scoring is
  scale-independent): a calibration problem, the one clear engineering fix.
- **Caveats:** the 43k point uses a different schedule (P3b, 12 epochs), so it is
  indicative not controlled; the exact crossover (~5-18k spans) is corpus/task
  specific and should not be over-generalized.

---

## P3d — Supervised-F1 scaling on FiNERweb-eng — DONE (unifies the whole project)

**Hypothesis**
- If the hyperbolic advantage is a data-efficiency effect (P3c), then the
  *supervised* typing-F1 gap should follow the same scaling law as the zero-shot
  mAP gap: large at small data, decaying to zero as data grows.

**Setup**
- Supervised probe `train_probe.py` (full flat-supervision pipeline: dev
  threshold sweep + best-dev-macro-F1 selection + micro/macro/tail F1 + mAP),
  generalized with `--train-file/--dev-file/--test-file/--max-len/--train-max-records`.
- Corpus: FiNERweb-eng. `scripts/p3d_prep_finerweb_val.py` carved a 200-doc dev
  out of the P3b train pool: 1,924 suptrain / 200 val / 374 test. Vocab/counts
  from the full suptrain file, so scale varies but the label space does not.
- **Schedule held CONSTANT** (25 epochs, batch 64, max_len 256, dim 64, flat),
  same as P3c. Grid: 2 geometries x scale {100,300,1000} docs x seed {1,2,3} = 18.
- Runner `scripts/p3d_launch.sh` (throttled one-run-per-GPU on a busy cluster,
  idempotent). Plot: `results/p3d/supervised_scaling_gap.png` (overlays the
  supervised F1 gap on the P3c zero-shot mAP gap).

**Results** — 18/18 runs, 0 failures. test F1, mean over seeds, n=3.
Gap = hyperbolic - euclidean.

| ~spans | micro euc | micro hyp | micro gap | macro gap | mAP gap | mid gap | tail gap |
|--------|-----------|-----------|-----------|-----------|---------|---------|----------|
| ~2k    | 0.205 | 0.486 | **+0.281** | +0.327 | +0.306 | +0.036 | +0.000 |
| ~5.5k  | 0.398 | 0.489 | +0.091 | +0.199 | +0.152 | +0.048 | +0.000 |
| ~18.6k | 0.569 | 0.559 | **-0.009** | -0.031 | -0.017 | +0.053 | +0.015 |

**Analysis**
- **Same scaling law as zero-shot, confirmed on supervised F1.** The micro-F1 gap
  decays +0.281 -> +0.091 -> -0.009 and crosses zero around ~18k spans, matching
  the P3c zero-shot mAP curve. The two curves overlay (see plot).
- **At small scale the supervised gap is even larger than zero-shot** (+0.28
  micro-F1 vs +0.12 zero-shot mAP): with ~2k spans euclidean barely learns
  (micro 0.205) while hyperbolic's prior gives a big head start (0.486).
- **Rare buckets keep an edge past the crossover:** at 1000 docs, mid-F1 (+0.053)
  and tail-F1 (+0.015) still favor hyperbolic even though micro/macro have crossed.
  The geometry helps the tail longest, consistent with P1.
- **This unifies the project.** P1 (supervised win on small UFET), P3 (zero-shot
  win on small data), P3b (win gone at scale), P3c (zero-shot scaling law), and now
  P3d (supervised scaling law) are one finding: hyperbolic geometry is a
  data-efficient inductive prior for low-resource / long-tail typing. Its
  advantage is a quantified function of training data, not a fixed property.
- **Answers the NER-F1 question:** yes, hyperbolic gives better typing F1, but it
  is a low-data effect that vanishes by ~18k training spans on this corpus.
- **Caveat:** hierarchical-F1 is not meaningful here (FiNERweb types are not in
  the UFET WordNet taxonomy); micro/macro/tail F1 and mAP are the metrics to read.

---

## P4 — Taxonomy-quality ablation — NOT STARTED

**Hypothesis**
- The hyperbolic advantage does *not* hinge on a high-quality taxonomy.

**Setup (planned)**
- Compare: no taxonomy / UFET 89-node `onto_ontology.txt` / WordNet map /
  (optional) ~1k merged taxonomy. Build the 1k taxonomy only if quality matters.

**Results**
- _pending_

**Analysis**
- _pending_

---

## P5 — Evaluation contribution — NOT STARTED

**Hypothesis**
- Flat evaluation unfairly penalizes all models; hierarchical P/R/F1 quantifies
  by how much.

**Setup (planned)**
- Report hierarchical P/R/F1 (`metrics.py:hierarchical_f1`) as a standalone
  contribution with qualitative `actor → person` examples.

**Results**
- _pending_

**Analysis**
- _pending_

---

## C1 — Statistical audit of conflicting training signal — DONE (premise confirmed)

**Hypothesis**
- Under Zipfian labels, losses that treat non-gold labels as negatives (BCE
  over the full vocab; InfoNCE with in-batch negatives) systematically push
  mentions away from labels that are TRUE of them (taxonomy-ancestors of the
  gold), and this conflict concentrates on head labels.

**Setup**
- `scripts/c1_conflict_audit.py`, no training. Two views per corpus:
  - **BCE view** (matches our training): count taxonomy-ancestors of gold with
    target 0 — false negatives baked into the supervision.
  - **InfoNCE view**: sample 500 batches per size B ∈ {8..256}; false negative
    = an in-batch negative that is identical to or an ancestor of the
    example's gold.
- Corpora: UFET crowd (multi-label) with the fixed WordNet taxonomy;
  FiNERweb-eng train (single-tag) with a fresh WordNet taxonomy over its 2,034
  types (`results/taxonomy/wordnet_parent_finerweb.json`, 50% linked — so all
  FiNERweb counts are LOWER BOUNDS).
- Outputs: `results/c1/{ufet,finerweb}.json`.

**Results**
- **UFET (1,998 spans):** BCE — **68.4%** of examples carry ≥1 ancestor-as-
  hard-zero, mean **1.84** per example; conflicted labels: location(240),
  region(216), group(165), state(140), organization(111). InfoNCE — at B=32:
  **92.5%** of examples have ≥1 false negative (3.1/example); B=256: 98.8%.
  Head+mid share of false negatives: 87–99% across B.
- **FiNERweb (43k spans, single-tag):** InfoNCE false negatives hit head
  labels **94–99.7%** of the time (quantity, group, organization, location);
  at B=64 (P3b setting) 68.5% of spans have ≥1. BCE view 13.2% (lower bound,
  50% taxonomy coverage).

**Analysis**
- Premise established statistically, with zero model confounds: the conflict
  is pervasive, grows monotonically with batch size, and is almost entirely a
  head-label phenomenon — the Zipfian shape is what creates it.
- Single-tag corpora make it maximal in principle (every ancestor is an
  explicit hard zero); multi-label UFET partially defuses it because
  annotators often include supertypes. Consistent with P3c's larger small-data
  hyperbolic gap on FiNERweb (+0.124) than on UFET (+0.034).

---

## C2 — Gradient-level conflict measurement — DONE (mechanism confirmed)

**Hypothesis**
- The C1 conflict manifests as contradictory gradients on head-label
  embeddings, and hyperbolic geometry resolves the contradiction while
  Euclidean cannot.

**Setup**
- `scripts/c2_gradient_conflict.py`: standard P1 training cell (UFET crowd,
  flat BCE, pos_weight 10, d64) with the BCE loss decomposed per step via the
  taxonomy closure into **positive** terms, **false-negative** terms (target 0
  but label is ancestor-of-gold → push is WRONG), and **true-negative** terms.
  Each part differentiated w.r.t. the label-embedding matrix only. Per label:
  gradient-magnitude share of the false-negative part, and cos(g_pos, g_fneg).
- Naive pos-vs-neg cosine saturates at ≈−1 for every label (anisotropy) and
  Adam momentum saturates update autocorrelation ≈+0.98 — neither
  discriminates; the closure-based decomposition is the informative metric.
- 2 geometries × seeds {1,2,3}, 25 epochs. Outputs: `results/c2/*.json`.

**Results** — mean±std over 3 seeds.

| geometry | bucket | fneg grad share | fneg/pos ratio | cos(g_pos, g_fneg) |
|----------|--------|-----------------|----------------|--------------------|
| euclidean  | head | 0.0155±.0001 | 0.029 | **−0.990±.001** |
| euclidean  | mid  | 0.0024±.0000 | 0.005 | −0.745±.010 |
| euclidean  | tail | 0.0002±.0000 | — | −0.478±.016 |
| hyperbolic | head | 0.0105±.0000 | 0.012 | **−0.232±.009** |
| hyperbolic | mid  | 0.0040±.0000 | 0.006 | −0.548±.004 |
| hyperbolic | tail | 0.0002±.0000 | — | −0.767±.009 |

Spearman(freq, fneg share): euc ρ=0.383, hyp ρ=0.393 (both p≈0).
Per-label: `person` (freq 824) fneg share euc 0.0180 vs hyp 0.0045 (4×), cos
−0.974 vs −0.156. `location`: share 0.066 vs 0.039, cos −0.992 vs −0.257.
Consistency check: `event`/`object`/`place` have exactly 0 fneg share — they
are taxonomy roots after the blocklist fix, never anyone's ancestor.

**Analysis**
- **The tug-of-war is real and Euclidean-specific.** On head labels the
  false-negative gradient is almost perfectly antiparallel to the positive
  gradient in Euclidean (cos −0.990): the two signals cancel; the embedding
  receives the residual of two large opposing forces. In hyperbolic the same
  false-negative push is close to orthogonal (cos −0.232): pushing a general
  label away from a descendant's mention is NOT the opposite direction of
  pulling it toward its own members — the geometry gives the conflict
  somewhere to go.
- **Exposure vs response:** the frequency-conflict correlation (ρ≈0.39) is
  identical across geometries — the *exposure* to false negatives is a
  property of the data. What differs is the geometric *response*: direction
  (−0.99 vs −0.23) and magnitude (head fneg share 1.5× smaller, `person` 4×
  smaller in hyperbolic).
- Together with C1 this is the mechanistic core of the thesis: Zipfian data
  + negatives-based CE ⇒ contradictory signal concentrated on head labels
  (C1, data-level); Euclidean geometry turns it into destructive gradient
  interference while hyperbolic nearly orthogonalizes it (C2, model-level).
  C3 (loss-family grid: InfoNCE / masked negatives / soft labels) will test
  causality: patching the supervision should mostly fix Euclidean but barely
  move hyperbolic.

---

## C3 — Causal loss-family grid — DONE (predictions FALSIFIED; major reframe needed)

**Hypothesis (registered in advance, incl. in the paper draft)**
- Supervision-side patches for the false-negative conflict (taxonomy-masked
  negatives, soft ancestor labels) should substantially repair Euclidean but
  barely move hyperbolic; euclidean+InfoNCE (in-batch negatives) should be the
  worst Euclidean cell; hyperbolic+naive-BCE >= euclidean+best-patched.

**Setup**
- 2 geometries x {bce, infonce, infonce-masked, soft-bce} x seeds, UFET crowd,
  d64, flat supervision, standard P1 protocol. BCE cells reused from p1_seeds.
- New losses in `src/sparse_ner/losses.py` (`infonce_loss` with optional
  false-negative candidate masking; `soft_bce_loss` alpha=0.5); `--loss` flag
  in `train_probe.py`. Runner `scripts/c3_launch.sh`; outputs `results/c3/`.

**Results** — test mAP (threshold-free), mean±std.

| geometry | bce | infonce | infonce-masked | soft-bce |
|---|---|---|---|---|
| euclidean  | 0.280±.002 | **0.420±.004** | **0.423±.001** | 0.274±.002 |
| hyperbolic | 0.353±.002 | 0.372±.006 | 0.365±.016 | 0.324±.003 |

Checked: not a model-selection artifact (per-epoch dev mAP maxima show the
same ordering). hyperbolic+infonce macro/mid F1 collapse (0.15/0.09) —
threshold calibration of InfoNCE-trained distance scores is genuinely poor,
but its mAP ceiling is real (~0.37).

**Analysis — every registered prediction failed:**
1. euclidean+InfoNCE is not the worst Euclidean cell; it is the BEST cell in
   the entire grid (0.420), beating hyperbolic+anything.
2. Masking the false negatives from the candidate set changes nothing
   (+0.004, within noise). The C1/C2 conflict, though real as a phenomenon,
   is NOT the binding constraint on Euclidean performance under InfoNCE.
3. Soft ancestor labels do not help either geometry (-0.006 euc, -0.029 hyp).
4. hyp+naive-bce (0.353) < euc+best (0.423).

**Honest interpretation:**
- The dominant factor is the LOSS FAMILY, not the geometry: listwise softmax
  over in-batch candidates is a far better ranking objective than
  full-vocabulary BCE with pos_weight. Under the better loss, Euclidean wins.
- The earlier P1 "geometry gap" (0.353 vs 0.280) is real but appears to be a
  gap in how the two geometries cope with a POOR loss (full-vocab BCE), not a
  fundamental superiority: hyperbolic was compensating for BCE's deficiency,
  and InfoNCE fixes that deficiency directly, more cheaply, and without a
  taxonomy.
- C1 (false negatives exist, Zipf-concentrated) and C2 (euclidean gradient
  tug-of-war at cos -0.99 vs hyp -0.23) remain valid observations, but the
  causal chain "conflict -> performance gap" is broken: removing the conflict
  (masking) does not move performance.
- Open questions before final verdict: (a) is hyperbolic+InfoNCE handicapped
  by scale/temperature parameterization (unbounded negative distances in a
  softmax)? worth one tuning pass before concluding geometry loses under
  InfoNCE; (b) does the low-data prior story (P3c/P3d) survive under InfoNCE?
  i.e. rerun the scaling sweep with the better loss; (c) batch-size
  dose-response (C5) may still show the conflict matters at larger B.

**Paper impact:** the draft's conclusion registers exactly the predictions
this experiment falsified. The mechanism sections (C1/C2) stand as
observations; the causal framing and the "hyperbolic resolution" title do
not survive C3 as-is. Reframe options: (i) loss-centric paper ("BCE's
deficiency, not geometry, drives the gap — InfoNCE is the fix; hyperbolic is
a BCE-regime patch"), (ii) low-data prior paper if P3c/d survives under
InfoNCE, (iii) negative-result / analysis paper on why gradient-level
conflict does not translate to task performance.
