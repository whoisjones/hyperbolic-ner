# Hyperbolic Bi-Encoders for Long-Tail Fine-Grained Entity Typing

## Idea

Fine-grained entity typing data (UFET + distant NER dumps) is **Zipfian**,
**noisy**, and **hierarchical** (`actor ⊂ artist ⊂ person`). A flat Euclidean
classifier treats types as independent and penalizes "person" when the gold
label is "actor". We test whether a **hyperbolic (Poincaré-ball) output space** —
which embeds trees with near-zero distortion and naturally places general types
near the origin, specific types near the boundary — improves rare/tail-type
performance and lets us supervise so that predicting a correct supertype is not
counted as wrong.

Both models are the **same span-level bi-encoder**; only the scoring geometry
differs:

```
mention span --encoder--> pool [start,end] --> project to D ┐
type name k  --label embedding (D)------------------------- ┤
                                                            v
              euclidean:  logit = s·cos(m, l) + b
              hyperbolic: logit = -s·d_ball(m, l) + b
```

Label embeddings are ordinary parameters (seeded from backbone encodings of the
type names) mapped into the ball only in the forward pass, so a standard Adam
optimizer suffices — no Riemannian optimizer or extra dependency.

## Layout

```
src/sparse_ner/
  geometry.py   Poincaré-ball ops + GeometryHead (euclidean | hyperbolic)
  model.py      BiEncoderTyper: span pooling + label embeddings + head
  data.py       Unified loader for UFET and distant-NER JSONL schemas
  taxonomy.py   Parent map, ancestor closure, ancestor-propagated supervision
  losses.py     BCE (+pos_weight) and geometry-agnostic distance-ranking loss
  metrics.py    UFET P/R/F1 + tail-stratified F1 + hierarchical F1
  trainer.py    Minimal train/eval loop
scripts/
  e0_build_vocab_taxonomy.py   E0: unified vocab + long-tail stats
smoke_test.py   End-to-end sanity check (run this first)
```

## Data (kept under /vol/tmp/goldejon)

- `sparse_ner/data/` — UFET crowd/distant/ontonotes + FewNERD (multi-label, `spans→type[]`)
- `sparse_ner/release/ontology/` — `types.txt` (10,331 UFET types), `onto_ontology.txt` (89-node coarse tree, taxonomy skeleton)
- `multilingual_ner/data/training_jsonl/` — finerweb, finerweb_translated, pilener, nuner, euro_glinerx (distant, single-tag, `spans_char→tag`)

## Experiment plan

**Phase 0 — establish the premise (no training)**
- **E0a** unified type vocabulary + long-tail curve (% types with ≤5 mentions). `scripts/e0_build_vocab_taxonomy.py`
- **E0b** build ~1k-node taxonomy: seed with `onto_ontology.txt`, attach rest via embed-cluster synonym merge + WordNet hypernyms. *(next)*

**Phase 1 — geometry go/no-go (no NER model)**
- **E1** fit Euclidean vs Poincaré embedding of the taxonomy tree; measure tree distortion + parent-retrieval mAP. Cheap gate on the whole premise.

**Phase 2 — model probe on one dataset (start: UFET crowd)**
- **E2** Euclidean vs hyperbolic bi-encoder, everything else fixed; report micro/macro F1 **stratified head/mid/tail**.
- **E3** flat BCE vs ancestor-propagation supervision, per geometry; report standard F1 + hierarchical F1.

**Phase 3 — scale & noise**
- **E4** add noisy distant data (PileNER/NuNER); test hyperbolic robustness to label noise.
- **E5** (optional) cross-lingual transfer.

## Running

```bash
source /vol/tmp/goldejon/.uv/envs/mm/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1
python smoke_test.py
python scripts/e0_build_vocab_taxonomy.py \
    --paths /vol/tmp/goldejon/sparse_ner/data/ufet_crowd_train.jsonl \
    --out results/e0
```
