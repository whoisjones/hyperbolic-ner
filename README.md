# Hyperbolic Representations for Long-Tail Hierarchical Labeling

Research code for whether **hyperbolic output geometry** helps models over
**Zipfian, hierarchical label spaces** (entity typing and beyond). The project
is mid-pivot: Phase 1's specific hypothesis was falsified and Phase 2 is being
designed. **Read [`docs/SYNTHESIS.md`](docs/SYNTHESIS.md) first** — it is the
current source of truth for where we are and why.

## Status (2026-07)

- **Phase 1 — symmetric hyperbolic bi-encoder: FALSIFIED.** A symmetric
  Poincaré-distance head beat Euclidean cosine *only under cross-entropy*; with
  a listwise in-batch loss (InfoNCE) Euclidean wins at every scale. Full record:
  [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md); archived code:
  [`archive/phase1_symmetric_bienc/`](archive/phase1_symmetric_bienc/);
  per-directory result map: [`results/INDEX.md`](results/INDEX.md).
- **What survived (carries into Phase 2):** the false-negative conflict audit
  (C1) and gradient tug-of-war measurement (C2), the noise-robustness and
  taxonomy findings, and the fair-comparison protocol.
- **Phase 2 — asymmetric, radius-supervised (in design).** Motivated by the
  Phase-1 diagnosis and by *Language Models as Hierarchy Encoders* (HiT,
  He et al. NeurIPS 2024): drop L2-normalization, use a circumscribed-ball
  manifold (no `expmap` saturation), supervise the radius directly (centripetal
  loss), and score membership asymmetrically. Our niche vs HiT: mention-level
  typing under Zipf, the C1/C2 conflict analysis, and **corpus frequency as a
  taxonomy-free radial prior**. See SYNTHESIS §4.

## Layout

```
src/hyperbolic_ner/     Core library (reusable across phases)
  geometry.py   Poincaré-ball ops + GeometryHead (euclidean | symmetric-hyp)
  model.py      BiEncoderTyper: span pooling + label embeddings + head
  data.py       Unified loader for UFET and distant-NER JSONL schemas
  taxonomy.py   Parent map, ancestor closure, ancestor-propagated supervision
  losses.py     BCE / InfoNCE (+masked) / soft-BCE / distance-ranking
  metrics.py    UFET P/R/F1 + tail-stratified F1 + hierarchical F1
  trainer.py    Minimal train/eval loop
train_probe.py          Main training harness (fair-comparison protocol) [live]
smoke_test.py           End-to-end sanity check — run first
scripts/                Reusable tools:
  build_taxonomy_wordnet.py   WordNet taxonomy w/ wrong-sense guards
  c1_conflict_audit.py        Conflict audit instrument (no training)
  c2_gradient_conflict.py     Gradient-decomposition instrument
  e0_build_vocab_taxonomy.py  Vocab + long-tail stats
  p3b_prep_finerweb.py / p3d_prep_finerweb_val.py   Data-split prep
  paper_figures.py            Figure generator
docs/                   RESULTS_LOG.md · SYNTHESIS.md · HANDOVER_phase1.md
archive/phase1_symmetric_bienc/   Falsified-phase launchers + zero-shot trainer
results/                Raw outputs per experiment (see results/INDEX.md)
```

## Data (`/vol/tmp/goldejon/`)

- `hyperbolic_ner/data/` — UFET crowd/distant/ontonotes + FewNERD (`spans→type[]`)
- `hyperbolic_ner/release/ontology/` — UFET `types.txt` (10,331), `onto_ontology.txt` (89-node tree)
- `multilingual_ner/data/training_jsonl/` — finerweb, finerweb_translated, pilener, nuner, euro_glinerx (distant, single-tag, `spans_char→tag`)

## Running

```bash
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
python smoke_test.py
```

Launch scripts run one job per GPU with `OMP_NUM_THREADS=8` (unthrottled
concurrency once caused 256-core oversubscription). GPUs are shared — check
`nvidia-smi` before launching.
