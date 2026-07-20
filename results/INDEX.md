# Results Index

Maps each `results/` subdirectory to its experiment, one-line finding, and
verdict. Authoritative narrative: [`../docs/RESULTS_LOG.md`](../docs/RESULTS_LOG.md).
Post-mortem: [`../docs/SYNTHESIS.md`](../docs/SYNTHESIS.md). Everything below is
**Phase 1 (symmetric hyperbolic bi-encoder)** unless noted. Launch scripts:
`archive/phase1_symmetric_bienc/launchers/`.

| dir | runs | experiment | finding | status |
|-----|------|-----------|---------|--------|
| `e0/` | — | Long-tail premise (no training) | 76% of types ≤5 mentions; top-10 = 51% mass | ✅ holds |
| `taxonomy/` | — | WordNet parent maps (UFET + FiNERweb) | wrong-sense-guarded hypernym taxonomy | ✅ reusable |
| `p1/` | 12 | Geometry probe grid (seed 42) | hyp > euc under BCE, +26% mAP, 2.7× mid-F1 | ⚠️ loss-bound (see C3) |
| `p1_seeds/` | 18 | P1 4-seed replication (key cells) | confirms P1 with tight σ | ⚠️ loss-bound |
| `p1_anc_v2/` | 24 | Ancestor supervision, fixed taxonomy | ancestor-prop underperforms flat in both geoms | ✅ holds |
| `p2/` | 42 | Controlled label-noise robustness | hyp robust to uniform noise, **fragile to sibling** noise | ✅ holds |
| `p3/` | 12 | Zero-shot held-out types (UFET) | pure-ZS hyp +44%; generalized ZS hyp loses (seen-bias) | ⚠️ loss-bound |
| `p3b/` | 6 | Zero-shot scale-up (FiNERweb) | UFET ZS win does NOT replicate at 43k spans | ⚠️ confound → P3c |
| `p3c/` | 18 | Data-scaling ablation (ZS) | clean scaling law; hyp gap crosses 0 at ~5–18k spans | ⚠️ under BCE only |
| `p3d/` | 18 | Data-scaling ablation (supervised F1) | same scaling law on supervised F1 | ⚠️ under BCE only |
| `c1/` | 2 | **Conflict audit** (no training) | 92.5% of examples carry false in-batch negs @B=32; Zipf-concentrated | ✅ **key insight** |
| `c2/` | 6 | **Gradient conflict** measurement | head-label cos(g_pos,g_fneg): −0.99 euc vs −0.23 hyp | ✅ **key insight** |
| `c3/` | 18 | **Loss-family grid** (causal test) | euc+InfoNCE beats all; masking conflict changes nothing | ❌ **falsifies causal claim** |
| `c3b/` | 12/18 | InfoNCE scaling (FiNERweb) | low-data hyp prior does NOT survive InfoNCE | ❌ **falsifies prior story** (running) |
| `presentation/` | — | Old slide deck + figures | Phase 1 talk assets | 🗄️ stale |

**Legend:** ✅ finding stands · ⚠️ true only in the falsified regime (BCE /
symmetric) · ❌ negative result that killed a Phase-1 claim · 🗄️ stale asset.

**Net:** the symmetric-hyperbolic performance thesis is dead (C3/C3b); the
conflict/gradient *analysis* (C1/C2) and the noise/taxonomy findings stand and
carry into Phase 2. Do not re-run any ⚠️/❌ cell to "double-check the win" — the
win was a BCE artifact; that question is settled.
