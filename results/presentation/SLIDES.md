# Weekly meeting talk: Hyperbolic geometry for long-tail entity typing
2-5 min. Explainer-first: build the setup, THEN show results.
Insert PNGs in Google Slides via Insert > Image > Upload from computer.
Figures are 16:9, sized to fill a slide.

One-sentence takeaway:
**Hyperbolic geometry is a data-efficient prior for long-tail / low-resource
entity typing. It helps most when training data is scarce; the advantage is a
scaling law that fades as data grows.**

---

## Slide 1 - What is the task?   [s1_task.png]
Say: "We do fine-grained entity typing. Given an entity mention in text, predict
all the types that describe it. One entity gets many types at once, and the types
form a hierarchy - soccer player is under athlete is under person. There are
10,000+ possible types and about three quarters are rare (<=5 examples). That
long tail is the whole problem."

## Slide 2 - What is our model?   [s2_model.png]
Say: "A simple bi-encoder. We encode the mention with BERT to a vector, encode
each type's name to a vector, and score by how close they are. The ONE thing we
change between the two models we compare is the geometry of that space:
euclidean flat space vs hyperbolic curved space."

## Slide 3 - Why would hyperbolic help?   [s3_geometry.png]
Say: "Hyperbolic space fits trees almost perfectly - general types sit near the
center, specific types near the rim, and there is exponentially more room as you
go out. So a rare, specific type can inherit its position from its parent instead
of being learned from scratch. That is the bet."

## Slide 4 - The hardest test: zero-shot   [s4_zeroshot.png]
Say: "We also test types the model has NEVER seen in training. At test time we
place a brand-new type using only its name, and check whether entities of that
type land near it. This is where a good geometry should pay off most."

## Slide 5 - Result 1: small data   [s5_result_smalldata.png]
Say: "With little training data (~2,000 examples) hyperbolic wins clearly - it
more than doubles F1 over euclidean. Same story on the tail types and on the
zero-shot unseen types."

## Slide 6 - Result 2: the scaling law   [s6_result_scaling.png]
Say: "But it is not a free lunch. If we grow the training set, the advantage
shrinks and disappears around ~18k examples - euclidean catches up because with
enough data it just learns the structure directly. Both the supervised and the
zero-shot gaps follow the same curve. So the honest conclusion is: hyperbolic
geometry is a data-efficient prior - it substitutes for data, and matters exactly
in the low-resource, long-tail regime we care about."

## Close (verbal, no slide)
"Next steps: check whether this needs a good taxonomy, fix the one setting where
hyperbolic still loses (ranking unseen vs seen types together), and frame the
hierarchical evaluation as its own contribution."

---

## If you only have 2 minutes: show slides 1, 2, 6.
## If you have 3-4 minutes: 1, 2, 3, 6 (add 5 for a concrete number).

## Numbers to have ready
- Long tail: 75.95% of types have <=5 mentions.
- Small data (~2k ex): micro-F1 hyperbolic 0.49 vs euclidean 0.20 (+0.28).
- Zero-shot (UFET): hyperbolic +44% mAP (0.111 vs 0.077).
- Scaling gap (supervised F1): +0.28 (2k) -> +0.09 (5.5k) -> -0.01 (18.6k spans).
- Crossover ~18k spans. All numbers are 3 seeds.

## Honesty notes (good to volunteer)
- This is entity TYPING, not full NER (span detection is separate, not done yet).
- The ~18k crossover is corpus/task specific - do not over-generalize the number.
- Generalized zero-shot (rank unseen AND seen types together) is the one setting
  hyperbolic still loses; it is a known, fixable calibration issue.

## Figure files (in this folder)
s1_task.png, s2_model.png, s3_geometry.png, s4_zeroshot.png,
s5_result_smalldata.png, s6_result_scaling.png
