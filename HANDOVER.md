# Handover: Hyperbolic Bi-Encoders for Long-Tail Entity Typing

## The research question

Fine-grained entity typing data is (a) **Zipfian** — most types have ≤5 training
examples, (b) **noisy** — largely distant/weak supervision, and (c) **implicitly
hierarchical** — `actor ⊂ artist ⊂ person`, but datasets annotate flat labels.

**Central hypothesis:** a bi-encoder that scores mention spans against label
embeddings in hyperbolic (Poincaré-ball) space outperforms the identical model
in Euclidean space, specifically on rare types — because hyperbolic geometry
embeds tree structure with low distortion (general concepts near the origin,
specific ones near the boundary), so rare types can inherit geometry from their
ancestors instead of needing to be learned from scratch.

**Coupled hypothesis on supervision & evaluation:** when the gold label is
`actor`, predicting `person` is not *wrong* — flat BCE training and flat F1
evaluation both punish it. We test ancestor-propagated supervision (expand gold
labels with their taxonomy ancestors) and hierarchical F1 (partial credit for
correct supertypes), and specifically whether hierarchy-aware supervision
**composes with hyperbolic geometry** in a way it doesn't with Euclidean.

## Setup (all in place, working)

- **Repo:** `/vol/fob-vol7/mi18/goldejon/sparse_ner` =
  github.com/whoisjones/hyperbolic-ner. Run `python smoke_test.py` first — it
  validates geometry math, both model variants, data loading, and metrics
  end-to-end.
- **Environment:**
  ```bash
  source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
  export PYTHONPATH=src HF_HUB_OFFLINE=1 NLTK_DATA=/vol/tmp/goldejon/nltk_data
  ```
  Encoder: `bert-base-uncased` (locally cached). No geoopt needed — label
  embeddings live in tangent space and are mapped onto the ball inside the
  forward pass, so plain AdamW works (`src/sparse_ner/geometry.py`).
- **The only knob between conditions** is `GeometryHead`: Euclidean = scaled
  cosine; hyperbolic = negative Poincaré geodesic distance. Everything else
  (encoder, pooling, data, schedule) is shared — keep it that way; the
  controlled comparison *is* the paper.
- **Data:**
  - UFET at `/vol/tmp/goldejon/sparse_ner/data/` (crowd = clean multi-label;
    distant/ontonotes = noisy, large). Schema: `spans → type[]`.
  - Distant NER dumps at `/vol/tmp/goldejon/multilingual_ner/data/training_jsonl/`
    (finerweb, finerweb_translated, pilener, nuner, euro_glinerx). Schema:
    `spans_char → tag` (single label).
  - `src/sparse_ner/data.py` normalizes both schemas to per-span examples.
- **Taxonomy:** `results/taxonomy/wordnet_parent.json` — mechanical WordNet
  hypernym linking over the 2,519 UFET-crowd types, with a frequency filter
  (parent must be ≥ as frequent as its child) that prunes wrong-sense chains
  like `person → cause → event`. Rebuild with
  `scripts/build_taxonomy_wordnet.py`.

## Fair-comparison protocol (non-negotiable, encoded in `train_probe.py`)

Cosine logits and negative distances calibrate completely differently, so any
fixed threshold biases the comparison:

1. Threshold is swept **per model** on dev via a quantile grid targeting
   average-predictions-per-example (geometry-agnostic calibration).
2. Always also report **mAP**, which is threshold-free — if a "win" appears in
   F1 but not in mAP, it is a calibration artifact.
3. Model selection = best dev macro-F1 epoch.
4. Every geometry claim needs a **dim sweep** (hyperbolic advantages
   classically live at low dim) and **multiple seeds**.

## Experiment ladder

### P1 — geometry probe (DONE, gate passed)
2 geometries × {flat, ancestor} supervision × dim {16, 64, 128} × 4 seeds on
UFET crowd. Runners: `train_probe.py`, `scripts/p1_launch_{grid,seeds}.sh`;
outputs in `results/p1*/`. This was the go/no-go for the whole thesis.

### P2 — controlled noise robustness (NEXT, start here)
**Hypothesis:** hyperbolic degrades more gracefully under label noise, because
realistic noisy labels are hierarchy-neighbors and the geometry keeps them
close.
**Method:** inject label corruption into UFET crowd *train* at
{0, 10, 30, 50}%, two modes: swap gold type for a taxonomy **sibling**
(realistic noise) and for a **uniform random** type (adversarial control).
Plot degradation curves per geometry.
Do **not** use "add PileNER" as the noise experiment — it confounds
distribution, scale, and noise.
**Implementation:** add `--noise-rate` / `--noise-mode` flags to
`train_probe.py`, corrupting train labels only, seeded.

### P3 — scale + zero-shot tail
**Hypothesis:** name-seeded label embeddings (`BiEncoderTyper._init_labels`)
plus hyperbolic geometry enable typing of **held-out/unseen tail types**,
where P1 showed pure supervision gets ~0 in any geometry.
**Method:** train on FiNERweb-eng (then +PileNER/NuNER), hold out a stratified
slice of tail types entirely, evaluate zero-shot ranking on them.
This is the most distinctive selling point — prioritize it over more UFET grid
cells.

### P4 — taxonomy-quality ablation
**Hypothesis to test:** the hyperbolic advantage does *not* hinge on a good
taxonomy (if it does, that is critical to know).
**Method:** compare no taxonomy / UFET's 89-node `onto_ontology.txt`
(in `/vol/tmp/goldejon/sparse_ner/release/ontology/`) / the WordNet map /
optionally a ~1k merged taxonomy. Only build the fancy 1k taxonomy if this
ablation says taxonomy quality matters.

### P5 — evaluation contribution
Frame hierarchical P/R/F1 (`src/sparse_ner/metrics.py:hierarchical_f1`) as a
standalone contribution: quantify how much flat evaluation *unfairly penalizes*
all models, with qualitative `actor → person` examples.

### P6 — optional, last
End-to-end NER (add a span proposer in front — geometry-irrelevant, keep it out
of the core claims) and/or cross-lingual transfer with the multilingual
FiNERweb dumps.

## Known open issues / warnings

1. **hyp-d128 training instability** (best epoch very early, then degrades) —
   likely distances saturating near the ball boundary; try a lower head LR or a
   max-norm constraint on label embeddings before trusting d128 numbers.
2. **True tail (≤5 examples) is unlearnable via supervision alone** at
   UFET-crowd scale — don't burn time trying to fix that inside P1/P2; it's
   P3's job.
3. Ancestor propagation **lowers flat F1 by construction** (predicted
   supertypes count as false positives) — always read it against hierarchical
   F1, never flat F1 alone.
4. Anything under `/vol/tmp/goldejon/multilingual_ner/finerweb_artifacts` is
   off-limits per prior guidance; the training data is the `training_jsonl`
   folders.
