# Audit

Dated log of editorial passes and verification runs. Newest first.
(P-JEPA's deeper records live in `docs/CRITIQUE.md`, `docs/HYPOTHESIS_RESULTS.md`.)

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
