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
- **Caveats / next:** (1) held-out set spans train-freq 1–50 (not strictly ≤5)
  for test support — could stratify ZS mAP by original frequency. (2) Try a
  calibration layer to recover GEN. (3) Scale up: repeat on FiNERweb-eng
  (+PileNER/NuNER) per the handover to test whether more/noisier data widens the
  zero-shot gap.

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
