# Audit

Dated log of editorial passes and verification runs. Newest first.
(P-JEPA's deeper records live in `docs/CRITIQUE.md`, `docs/HYPOTHESIS_RESULTS.md`.)

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
