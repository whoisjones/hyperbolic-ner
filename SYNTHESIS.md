# Synthesis — What We Learned, Why the Approach Stalled, Where to Go

Written 2026-07-20, after C3/C3b falsified the project's causal claims.
Sources: RESULTS_LOG.md (E0 → C3b). This document is the honest reading.

## 1. What we established (findings that stand)

**F1 — The regime is real.** Fine-grained typing labels are severely
Zipfian (76% of types ≤5 mentions; top-10 types = 51% of mass) and
implicitly hierarchical, while annotation is hierarchy-incomplete. (E0)

**F2 — The supervision contradicts itself, at the head.** Under
full-vocab BCE, 68% of UFET examples penalize ≥1 true-but-unannotated
ancestor; with in-batch negatives at B=32, 92.5% of examples carry false
negatives; ~90% strike head/mid labels; worse in single-tag corpora. (C1)

**F3 — The contradiction is visible in the gradients, and the geometry
changes its shape.** On head labels, positive and false-negative gradient
components are antiparallel in Euclidean (cos −0.99) and near-orthogonal
in hyperbolic (−0.23). Exposure identical (ρ≈0.39 both), response
differs. (C2)

**F4 — Under full-vocab BCE, symmetric-hyperbolic beats
symmetric-Euclidean**, +26% rel. mAP, 2.7× mid-frequency F1, d16-hyp >
d128-euc; the advantage decays with training data (crossover ~5–18k
spans) and is largest in low-resource / zero-shot settings (+44% pure ZS
on UFET). (P1, P3, P3c, P3d)

**F5 — The hyperbolic advantage is loss-bound, not fundamental.** With a
listwise in-batch objective (InfoNCE), Euclidean-cosine jumps to the best
result in every tested cell (UFET 0.420 vs hyp-best 0.372; FiNERweb @2k
spans 0.634 vs 0.527 interim), including the low-data regime where the
BCE gap was largest. Masking the false negatives changes nothing
(+0.004); soft ancestor labels help nothing. (C3, C3b)

**F6 — Structured noise is hyperbolic's specific weakness.** Sibling-swap
noise (hierarchy-local) erodes symmetric-hyperbolic faster than
Euclidean; uniform noise barely touches it. (P2)

**F7 — Method assets worth keeping:** the false-negative audit (C1), the
gradient-decomposition instrument (C2), the calibration-free comparison
protocol, the WordNet taxonomy builder with wrong-sense guards, and the
scaling-sweep harness.

## 2. The honest conclusion

The performance story we chased — "hyperbolic output space is inherently
better for Zipfian typing" — is falsified **in the form we tested it**:
a symmetric Poincaré *distance* head bolted onto an unchanged,
LayerNorm'ed transformer, trained with pointwise/listwise CE. Everything
symmetric-hyperbolic "fixed" was compensation for full-vocab BCE being a
bad ranking objective. C1/C2 documented a real pathology, but removing it
surgically (masked negatives) moved nothing: the binding constraint was
the loss, not the conflict.

## 3. Why this specific instantiation was doomed (diagnosis)

**D1 — Symmetric distance cannot represent entailment.** "actor ⊂ person"
is an order relation, not a proximity relation. Any metric — cosine OR
geodesic — forces "person close to actor-mentions" and "person distinct
from actor" into the same scalar. The C2 tug-of-war is a symptom of
scoring hierarchy with a symmetric function; hyperbolic distance merely
bends the contradiction into orthogonality instead of resolving it.

**D2 — The radial degree of freedom was never populated.** Hyperbolic
space encodes hierarchy in the *radius* (general = near origin, specific
= near boundary). But (a) transformer features pass through LayerNorm,
which collapses norm variance, and (b) cosine-style pipelines L2-normalize
— so features land at ~fixed radius. At fixed radius, entailment-cone
apertures collapse to a constant and the hierarchy-encoding capacity of
the space is unused. We attached a radial geometry to an encoder that is
architecturally prevented from producing radial information.

**D3 — Labels were points, not regions.** A general type covers a set of
mentions; a point-embedding forces one location. Cones / boxes / balls
(regions with containment) natively express "person's region contains
actor's region" — and dissolve the C1 false-negative problem *by
construction*: no push away from person is needed for an actor-mention,
because containment is not exclusion.

## 4. The reframed research direction

**New hypothesis.** Hierarchical, Zipfian label spaces need an
*asymmetric, radially-informed* output geometry: labels as entailment
cones (or regions) whose position/aperture reflects generality, mentions
as points scored by *containment*, with the encoder given an explicit
radial channel (bypassing LayerNorm) so generality can be expressed.
Zipf becomes the radial prior: corpus frequency ≈ generality ≈ radius.

Under this hypothesis, our negative results are predictions, not
embarrassments: symmetric-hyperbolic had to fail (D1–D3), sibling noise
had to hurt (siblings are near-identical under symmetric distance but
disjoint as cones), and InfoNCE had to help Euclidean (it fixes the loss,
not the representation).

**Candidate tasks (structure explicit, Zipfian, asymmetric evaluation):**
- Hierarchical multi-label text classification with gold DAGs: RCV1-v2,
  WOS, Amazon product categories → exact audits, no WordNet noise.
- Taxonomy completion / expansion: WordNet subtrees, SemEval-2016 Task 13,
  public product taxonomies → the task IS the hierarchy.
- KG link prediction on hierarchy-heavy relations (WN18RR): published
  low-dim hyperbolic wins (MuRP, RotH) exist — the one setting where
  hyperbolic demonstrably works is relational/asymmetric, which supports
  the reframing.
- Customer-intent hierarchies (shallow but industrially motivated).
- Entity typing (UFET/FiNERweb) stays as the in-the-wild showcase, not
  the flagship.

**Probe ladder (cheap → expensive):**
- **N1 (day):** encoder-mismatch measurement — variance of mention-feature
  norms pre/post LayerNorm and MI(norm, label depth/frequency) on real
  data. Quantifies D2; motivates the radial channel with a number.
- **N2 (day):** structure-only probe on a gold taxonomy (RCV1): symmetric
  euc / symmetric hyp / Euclidean order-embeddings / hyperbolic cones on
  transitive-closure reconstruction + held-out edge prediction. Sanity
  gate that asymmetry, not curvature alone, is what pays.
- **N3 (week):** minimal asymmetric model — encoder with a
  direction+radius head (radius read off pre-LayerNorm features), labels
  as cones with depth/frequency-tied aperture, containment-energy loss +
  InfoNCE over candidates. Benchmark against the C3 champion
  (Euclidean-cosine + InfoNCE) on RCV1 + UFET. **This is the new go/no-go:
  if cones cannot beat the strong symmetric baseline on a gold-taxonomy
  task, the thesis is dead in its strong form too.**
- **N4:** breadth (taxonomy completion, WN18RR slice, product data) only
  after N3 passes.

**Kill criteria (registered now):** N2 must show the asymmetric scorers
dominate symmetric ones on structure; N3 must beat euc+InfoNCE on at
least mid/tail slices at matched dim, with the radial channel ablation
showing it matters. Two failures = abandon the geometry thesis entirely
and publish the loss-centric analysis paper (C1/C2/C3 already support it).
