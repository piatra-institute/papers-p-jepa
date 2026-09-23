# Audit

Dated log of editorial passes and verification runs. Newest first.
(P-JEPA's deeper records live in `docs/CRITIQUE.md`, `docs/HYPOTHESIS_RESULTS.md`.)

## 2026-09-23 — prose revision

Prose rewritten against the house standards. Headings: Abstract; 1 Introduction (was Reframing); 2 Intervention-Sufficient Representations; 3 Augmentations as Loss Terms; 4 NumPy JEPA Toy on Dishworld; 5 Preregistered Hypothesis Tests (H1 Obstruction Gate; H2 Active Versus Entropy Probing; H3 Trained Encoder Versus Frozen Random Projection; H4 Cellular Sheaf Versus Scalar Cover; H5 JEPA Augmentations on the Toy); 6 Conjectured Typology and Priority Order; 7 Limitations (absorbs "What this paper is not"); 8 Conclusion; 9 Reproducibility.
Tic counts before -> after: "this paper/the paper" 13 -> 0; inline ", not X" 7 -> 0; negate-pivots 3 -> 0; "rather than" 3 -> 0; sentence-initial "This is" 4 -> 0; voice roadmap warn removed.
Numerical corrections (checked against simulation/output/experiments/*.json):
  - Dishworld JEPA-toy contexts are 6-dimensional (4 sensor + 2 one-hot visual; data.CONTEXT_DIM = 6), stated as 11-dimensional. Text and the jepa_toy/__init__.py docstring corrected.
  - H5 +bisim mean score 0.46153 was printed as 0.461; corrected to 0.462 (paper and docs/HYPOTHESIS_RESULTS.md).
  - H4 coboundary energy after gluing 0.034548 was printed as 0.034; now 0.0345. Mean edges 8.95 (was 8.9) and mean dim H^0 9.85 (was "approx 9.8"), stated at their exact two-decimal values.
  - H3 score CI upper bound 0.0045 was printed as +0.005; now +0.0045.
  - Removed the claim of an "8 generated JSON artifacts" count (the experiments write five) and an unsourced "adjacent literature suggests ~0.5-1%" gain for active masking; removed "possibly in the JEPA-adjacent literature" as an unsupported priority claim.
  - docs/HYPOTHESIS_RESULTS.md also carried a stale H4 unsafe-rate CI [+0.0018, +0.0025]; the JSON gives [0, 0]; corrected there.
All other prose numbers match the JSONs (H1 obstruction 0.154-0.255 vs threshold 0.06; H2 +0.0130 [+0.0094, +0.0166], 41/50; H3 0.802/1.000 vs 0.800/0.994; H4 0.798 vs 0.802, CI [-0.0050, -0.0036]; H5 table; base JEPA range 0.443-0.797).
Grid audit: no grid-derived thresholds; bisimulation weight 0.3 is a fixed configuration value. No simulation code changed beyond the docstring; generated outputs are gitignored in this repo.
The References preamble describing the audit of an earlier reference list was removed; bibliography entries unchanged. Previously uncited entries now cited: LeCun 2022, Assran 2023, Bardes 2024, Pearl 2009, Ferns 2011, Zhang 2021, Ames 2019, Hansen & Ghrist 2019, Robinson 2017, de Haan 2019. Friston 2010 and Ross 2011 remain uncited.

## 2026-06-13 — voice reform

Voice-reform editing pass to remove AI-writing tells (house voice.md).

Lexical density: before — genuinely 2; tricolon proxy 14; plus "the whole point" pet-phrase and one negate-pivot. After — genuinely 2; tricolon proxy 12.

Changes:
- Fixed eight inline-contrastive (", not Z") and one negate-pivot construction into positive declaratives: abstract ("posterior-weighted variance, not a coboundary" -> "in place of a coboundary"; "directional signals... not quantitative rankings" -> "rather than quantitative rankings"); §3 ("a sampler, not a loss" -> "acts as a sampler rather than a loss"); §5 H2 ("underpowered, not wrong" -> "reached the right direction with too few seeds to resolve it"); §5 H3 ("by clustering... not by gradient training" -> "gradient training adds nothing measurable"); §6 bullet ("works on continuous overlapping data, not on categorical regimes" -> "it works on... and fails on..."); §6 close ("a priority order, not an architecture" -> "with a priority order over augmentations"); §7 ("works... not as evidence that the sheaf or 'neural' framings work" -> separated into a positive sentence plus "They say nothing about whether..."); §9 opening triple-negation ("It is not a new architecture. It is not a foundation model proposal." -> "This paper proposes neither a new architecture nor a foundation model.").
- Retitled generic §7 "Limits" to "What the toy can and cannot show". The §1 prose line "§7-8 are limits and reproducibility" still reads correctly as a description. No "§7" heading-style cross-references needed fixing.
- Removed pet-phrase "The whole point of §3's..." in §7 ("exist because the real test is at V-JEPA scale").
- Distinctive closing "What this paper is not" (§9) kept; its negative-framing identity preserved while the mechanical opening run was recast.
- Hard-wrapped style of this paper preserved (edits matched existing line wrapping; no reflow). No numbers, citations, math, code blocks, or tables touched. Math arrows remain escaped ($\to$); build emits no missing-char.

Verify: voice 0 errors, 0 warns; refs n/a (paper uses bullet-style references the refs tool does not parse, pre-existing); claims claim-ledger present, 24 verification files, reconciled; build clean (0 missing-char); check => PASS.

## 2026-05-29 — upgrade pass (Group A)

Scope: §5 led with the H1–H5 verdict structure so the reader sees the five
results before the prose unpacks each.

Changes:
- Added a 5-row verdict table at the top of §5 (hypothesis · verdict · evidence),
  numbers matching the per-hypothesis prose and the claim ledger.

Verification: voice 0 errors; claim-ledger present (24 verification files);
build clean, 10 pages; check => PASS. (Title was set to "P-JEPA: JEPA
Augmentations from Embodied and Causal Mathematics" in the prior pass.)

## 2026-05-29 — workspace alignment + voice cleanup

Scope: bring P-JEPA to the workspace publication bar.

Changes:
- metadata.yaml: real title + abstract synced from `paper/PAPER.md` (had been a
  `P-JEPA` placeholder); repo wired to `papers-p-jepa`.
- Voice: removed all 18 em-dashes. Prose-rhetorical pivots restructured to
  periods/commas; the H1–H5 heading label dashes became colons; the H5 table's
  baseline empty cells became `n/a`; the two paired-appositive lists (§7 sheaf
  components, §9 infrastructure) became parentheticals.
- Fixed the literal `→` (U+2192) glyph in §5/H4 (`0.354 → 0.034` → `to`), which
  Palatino cannot render.

Verification:
- voice: 0 errors (9 review-candidate warns: the "What this paper is not"
  scope-list and developed `, not` contrasts — kept deliberately).
- build: clean, **10 pages**, zero missing-character warnings. (The previously
  committed PDF was a stale 28pp artifact from an older draft.)
- claims: claim-ledger present (`docs/CLAIM_LEDGER.md`, 24 verification files).
- check => PASS.

Outstanding: not yet on the web papers page; GitHub repo still private.
