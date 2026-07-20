# Archive — Phase 1: Symmetric Hyperbolic Bi-Encoder (FALSIFIED)

This directory holds the launch scripts and the zero-shot entry point for the
project's **first research phase**, which was falsified by experiments C3/C3b
(July 2026). Kept so we (a) do not repeat these runs and (b) retain the
insights. **Full narrative: [`../../docs/RESULTS_LOG.md`](../../docs/RESULTS_LOG.md);
post-mortem + pivot rationale: [`../../docs/SYNTHESIS.md`](../../docs/SYNTHESIS.md).**

## What Phase 1 tested

A span-level bi-encoder scoring mention vectors against label embeddings, where
the *only* variable was the output geometry: **Euclidean cosine** vs **symmetric
Poincaré geodesic distance** (`GeometryHead` in `src/hyperbolic_ner/geometry.py`),
trained with cross-entropy. Hypothesis: symmetric hyperbolic distance is
inherently better for Zipfian, hierarchical entity typing.

## The verdict (why this phase is archived, not deleted)

- **Under full-vocab BCE, symmetric-hyperbolic beat symmetric-Euclidean**
  (+26% rel. mAP, ~2.7x mid-frequency F1) — P1, and this *decayed with data*
  (P3c/P3d scaling law, crossover ~5–18k spans).
- **But the advantage was loss-bound, not fundamental (C3/C3b).** With a
  listwise in-batch objective (InfoNCE), **Euclidean cosine becomes the best
  model at every scale**, including low-data. Surgically removing the C1/C2
  false-negative conflict (masked negatives) changed nothing. The symmetric
  hyperbolic "win" was compensation for BCE being a poor ranking objective.
- **Standing insights (still valid, reused going forward):** the false-negative
  conflict audit (C1), the gradient tug-of-war measurement (C2: cos −0.99 euc
  vs −0.23 hyp on head labels), the fair-comparison protocol, and the WordNet
  taxonomy builder with wrong-sense guards.
- **Diagnosis of why symmetric-hyperbolic had to fail** (SYNTHESIS §3): symmetric
  distance cannot encode entailment; LayerNorm/L2-norm collapses the radial
  degree of freedom; `expmap0`/tanh saturates points to the boundary (visible
  as decision thresholds drifting to −76 in C3b). These directly motivate
  Phase 2 (HiT-style asymmetric, radius-supervised, circumscribed-ball manifold).

## Contents

- `launchers/` — the per-experiment orchestration scripts (`p1_*`, `p2_*`,
  `p3*`, `c2_launch`, `c3_launch`) and their stdout logs. Each `cd`s to the
  repo root and calls `train_probe.py` / `train_zeroshot.py`.
- `train_zeroshot.py` — zero-shot held-out-type protocol (P3 series).
- The main training harness `train_probe.py` stays at the repo root: its
  fair-comparison protocol (dev threshold sweep, best-dev selection, metric
  suite, `--loss`/`--noise`/scaling flags) is reusable and Phase 2 extends it.
  `c3b_scaling_infonce.sh` also stays at root-level `scripts/` because it was
  still running at archive time.

## Reproduce (if ever needed)

Raw outputs are in `results/<experiment>/` (see `results/INDEX.md`). Scripts are
runnable in place; they use absolute repo paths.
