# MPNNKit interactive-design API — design

- **Date:** 2026-07-20
- **Status:** Approved (brainstorm) — ready for implementation planning
- **Repo:** `proteinmpnn-ios` (`MPNNKit` package)
- **Drives:** RayMol issue #217 — *Interactive on-device design tool: logit-colored backbone + per-position AA editing with live repack*

## 1. Context & decomposition

RayMol #217 asks for a **stateless, interactive protein-design tool inside the RayMol viewer**,
powered on-device by MPNNKit: color the backbone by per-residue design confidence, hover a
residue for its 20-AA probability vector, edit residues/selections from an amino-acid palette,
re-run design/repack on-device, and spawn each result as a new auto-aligned object.

That feature decomposes into independent sub-projects, each with its own spec → plan →
implementation cycle:

- **Phase 1 (this spec):** the MPNNKit public-API additions the feature needs — scoring
  (logits), (partial) redesign with fixed positions + palette constraints, and standalone
  side-chain repack. Lives entirely in `proteinmpnn-ios`. No RayMol code.
- **Phase 2 (later spec, in RayMol):** the viewer integration — a "Design" tool mode, per-residue
  confidence coloring + SwiftUI legend, the 20-AA hover/tap popover, the selection→palette
  redesign flow, and "each result = a new superposed object." Built against the finished Phase-1
  API. All UX and orchestration decisions (including the undo model) belong to Phase 2.
- **Separate issue:** "Predict" (structure prediction, ESMFold/AF2). A different engine that does
  not run on iPhone today — out of scope for both phases.

This ordering ("MPNNKit first") was chosen deliberately: it keeps MPNNKit a pure, verifiable
inference engine and defers every UX/composition question to Phase 2, where the coordinates and
the user already live.

## 2. Goals & scope

**In scope (Phase 1):**

1. `score()` — per-residue log-probabilities `[L,21]` under a selectable probability semantics,
   plus a convenience per-residue log-prob of the *current* amino acid. Powers the confidence
   coloring and the hover popover.
2. `design()` — (partial) redesign with **fixed positions**, **per-position bias**, and
   **per-position omit**. Powers selection-based redesign with the rest held fixed, and the
   palette semantics (hard-assign vs palette-restricted).
3. `repack()` — standalone side-chain repack for a fixed backbone + sequence, returning a PDB
   string plus the packer's per-atom confidence. Powers the "assign an AA → repack → rescore"
   loop.
4. Parity coverage for all of the above against the reference PyTorch LigandMPNN/ProteinMPNN.

**Invariants (must not regress):**

- **MLX-free public boundary.** Coordinates enter as `SIMD3<Float>`; everything returns as
  `String` / `[[Float]]` / `[Int]`. The host never imports MLX. (README's core promise;
  `MPNNModel.swift:1-3, 52`.)
- **`run()` stays and stays bit-stable.** It is re-expressed as a thin wrapper over
  `design()` + `repack()`, guarded by the existing greedy + `seed=0` parity test so the refactor
  cannot perturb the design hot path.
- **`.mpnnpack` unchanged.** No new weights, no `Package.swift` change — every capability already
  lives in the loaded model.

**Out of scope (Phase 1):** any RayMol code; structure prediction ("Predict"); partial-encode
optimization; iOS memory chunking; ligand-aware design.

## 3. Relevant internals (grounded)

All capabilities exist internally today; Phase 1 is mostly *exposure*, not new modeling.

- `MPNNModel.run()` (`MPNNModel.swift:57-97`) builds `X[1,L,4,3]`, `mask`, `Ridx`, `chainLabels`,
  a **random decode `order`**, and hardcodes `chainMask = ones`, `Snative = zeros` (all-Ala). It
  calls `featuresDesignE → encodeDesign → decodeSequence → (optional) repackFull → PDBWriter` and
  **discards the logits** (`let (S,_,_) = decodeSequence(...)`).
- `decodeSequence` (`DesignModel.swift:107-168`) already returns `(S, logits[B,L,21],
  logProbs[B,L,21])` and already implements fixed positions: per position `t`,
  `St = St*cmT + SnT*(1-cmT)` with `cm = mask * chainMask` (line 129, 156-158) — so
  **`chainMask=0` forces the native token `Snative[t]`**. The autoregressive state
  `hS[t] = W_s[St]` (line 161) feeds the chosen token forward, so a position's logits are
  conditioned on the residues decoded earlier in `order`.
- Sampling uses `logits[..., 0..<20]` (line 151/153) — index 20 (`X`) is never sampled.
- `repackFull` (`RepackLoop.swift:42-81`) is already a pure
  `(backbone X, sequence S) → (atom14, atom14Mask, bFactors)` function; `bFactors` is the packer's
  von-Mises mixture log-prob per atom slot (a side-chain confidence). `run()`'s repack branch
  (`MPNNModel.swift:88-93`) is exactly `repackFull` + `PDBWriter.write`.
- The only net-new *modeling* is per-position **bias/omit** — a `[L,21]` addend applied to
  `logits` before the sampling switch (`DesignModel.swift:146-148`).

## 4. Public API surface

All additions on `MPNNModel`. The existing `Residue`, `Options`, `Result`, `init`, and `run`
stay (see §6 for `run()`'s rewrap and the optional `Result.logits`).

```swift
// Index → 1-letter AA (index 20 = 'X'); lets the host read the [L,21] columns.
public static var alphabet: [Character] { Array("ACDEFGHIKLMNPQRSTVWYX") }

// ---------- score() : per-residue confidence (coloring + hover popover) ----------
public enum ScoreMode {
    /// logP(AA_i | structure + residues earlier in a seeded/canonical decode order).
    /// One decode pass, cheap. Order-dependent (pinned via `seed`). DEFAULT.
    case conditional
    /// logP(AA_i | structure only) — no sequence conditioning. Order-independent;
    /// most comparable across the parent and every redesign child.
    case unconditional
    /// logP(AA_i | structure + ALL other native residues). Order-independent;
    /// most faithful to "how well each residue fits given the rest"; costlier.
    case leaveOneOut
}

public struct ScoreResult {
    /// L × 21 natural-log probabilities; rows in input residue order, columns in `alphabet` order.
    public let logProbs: [[Float]]
    /// Per-residue log-prob of the CURRENT AA (present iff `sequence` was supplied).
    /// This is the scalar Phase 2 maps to the color ramp.
    public let currentAALogProb: [Float]?
}

/// Score a backbone (optionally its current sequence) without changing it. RNG-free.
public func score(_ residues: [Residue],
                  sequence: [Int]? = nil,      // alphabet indices, length L; required for .conditional/.leaveOneOut
                  mode: ScoreMode = .conditional,
                  seed: UInt64? = 0) throws -> ScoreResult

// ---------- design() : (partial) redesign, palette-constrained ----------
public struct DesignOptions {
    public var temperature: Float = 0.1
    public var seed: UInt64? = nil
    /// Positions (input-order indices) held fixed at native identity (chainMask = 0 there).
    public var fixedPositions: Set<Int> = []
    /// Native/fixed identities (alphabet indices), length L; required iff `fixedPositions` non-empty.
    public var nativeSequence: [Int]? = nil
    /// Per-position additive logit bias, L × 21 (aligned to `alphabet`); added before sampling.
    public var bias: [[Float]]? = nil
    /// Per-position disallowed AAs (alphabet indices) → those logits set to −inf.
    public var omit: [Set<Int>]? = nil
    public init() {}
}

public struct DesignResult {
    public let sequence: String     // 1-letter, input order
    public let indices: [Int]       // alphabet indices, input order
    public let logits: [[Float]]    // L × 21 decode logits
}

public func design(_ residues: [Residue], options: DesignOptions = DesignOptions()) throws -> DesignResult

// ---------- repack() : side chains only, fixed backbone + sequence ----------
public struct RepackResult {
    public let pdb: String              // repacked all-atom structure, re-loadable as a new object
    public let atomConfidence: [[Float]]  // L × 14 packer von-Mises log-prob per atom slot
}

public func repack(_ residues: [Residue], sequence: [Int]) throws -> RepackResult
```

### Mapping to #217's four capabilities

- **(a) scoring for coloring** → `score(mode:)`; `currentAALogProb` is the color scalar.
- **(b) partial redesign, rest fixed** → `design(fixedPositions:nativeSequence:)`.
- **(c) repack-only + rescore** → `repack()` then `score()` (the "rescore" is a second `score`
  call; the host composes).
- **(d) palette semantics** → hard-assign = fix the position + set that AA in `nativeSequence`;
  palette-restricted = `bias` / `omit`.

### Decisions recorded

- **`ScoreMode` is an enum (support several).** `score()` exposes `conditional` /
  `unconditional` / `leaveOneOut`; the trio expresses *how much context* each position sees
  (partial-order / none / all-others). Default `.conditional` — one cheap pass, literally "log-prob
  of the current AA", deterministic when seeded.
- **Composable primitives, not task-oriented ops.** MPNNKit exposes `score`/`design`/`repack`;
  RayMol (Phase 2) orchestrates the edit→repack→score loop. This keeps UX/composition (incl. the
  still-open undo model) out of the engine.
- **`bias`/`omit` indexed by the full 21-wide `alphabet`.** Index 20 (`X`) is never sampled, so a
  bias/omit entry on it is a harmless no-op — one indexing scheme everywhere.

## 5. Internal realization

**Guiding constraint:** touch `decodeSequence`'s core loop as little as possible so the design
parity stays bit-stable.

- **`design()`** — generalizes `run()`'s setup: `chainMask` = ones with `0` at `fixedPositions`;
  `Snative` from `nativeSequence` (else zeros); seeded `order` from `options.seed`. The **only edit
  to `decodeSequence`** is a nil-guarded `bias`/`omit` addend applied to `logits` immediately after
  `linear(w,"W_out",…)` (`DesignModel.swift:146`) and before `logp`/sampling. When `bias == nil`
  and `omit == nil` the op is skipped and the path is byte-identical to today. Returns
  `sequence`/`indices` from `S`, `logits` from the existing `logitsOut`.
- **`score()`** — no sampling; RNG-free; returns `logProbs` (the existing `logp`):
  - **`.conditional`** — *no new code*: call `decodeSequence` with `chainMask = zeros` +
    `Snative = sequence`. Every position is forced to native (`cmT = 0`), so `hS` accumulates the
    native tokens and `logProbs[t]` is the teacher-forced conditional. `currentAALogProb[i] =
    logProbs[i][sequence[i]]`.
  - **`.unconditional`** — a new sibling decode that drops the autoregressive term (`hS ≡ 0`,
    backward mask off): one parallel pass, order-independent. Mirrors the reference
    `unconditional_probs`-style path.
  - **`.leaveOneOut`** — a new sibling decode with a non-causal, self-excluded attention (native
    `hS` for all `j ≠ i`): one pass, order-independent. Mirrors the reference `conditional_probs`
    path.
  - `.unconditional` and `.leaveOneOut` are added as **siblings**, leaving the design loop
    untouched. The PyTorch reference is the parity oracle for each.
- **`repack()`** — `run()`'s repack branch minus design: `sequence:[Int] → S`,
  `repackFull(…, numSteps: 3)`, `PDBWriter.write(...)` → `pdb`, and `bFactors → atomConfidence`
  (L × 14).

## 6. `run()` rewrap & data flow

- **`run()` = `design()`** (all-free options, same `seed`/`order`) → if `repack`,
  `repack(design.indices)` → `Result`. Free-design with the same seed reproduces today's exact RNG
  call sequence, so this is bit-identical; the regression test is the proof. `Result` optionally
  gains `logits` (from the `design()` call it now delegates to).
- **fp32 featurization** preserved (internal; `featuresDesignE` — critical for the kNN graph).
- **Decode order:** `design` uses the seeded random order (as today). `score(.conditional)` uses a
  deterministic order from `seed` (default 0) so scores are reproducible; `.unconditional` /
  `.leaveOneOut` are order-independent (seed ignored). Documented on the mode.
- **MLX → Swift boundary:** `MLX.eval` then `.asArray(Float.self)` reshaped to `[[Float]]`. No MLX
  type escapes.
- **Memory:** inherits `decodeSequence`'s per-step `eval` that bounds the lazy graph
  (`DesignModel.swift:165`).
- **Threading:** all methods synchronous; the caller runs them off the main thread (README) —
  unchanged.

## 7. Error handling

A new typed error, thrown eagerly before any MLX work:

```swift
public enum MPNNInputError: Error {
    case emptyResidues
    case sequenceLengthMismatch(expected: Int, got: Int)   // score/repack sequence, nativeSequence
    case sequenceRequired(ScoreMode)                       // .conditional / .leaveOneOut need a sequence
    case nativeSequenceRequired                            // fixedPositions set but nativeSequence nil
    case indexOutOfRange(Int)                              // fixedPositions / AA index ∉ valid range
    case biasShapeMismatch(expected: (Int, Int), got: (Int, Int))
}
```

AA indices validated in `0..<21`; `fixedPositions` in `0..<L`; `bias` exactly `L × 21`. The
`.mpnnpack` load path (`verifyHashes`, sha256) is unchanged.

## 8. Testing & parity strategy (TDD)

Follows the repo's established two-tier pattern (`port/*.py` oracle ↔ MLX-Swift, then Swift
package tests), using the local `ProteinMPNN` / `LigandMPNN` checkouts as ground truth.

**Tier 1 — Python oracle (`port/`).**

- New `port/capture_score.py` + `port/validate_mlx_score.py` → per-position logprobs for all three
  modes on a fixture (mirrors `capture_design.py` / `validate_mlx_design.py`).
- Extend `port/validate_mlx_design.py` → fixed-positions, `bias`, and `omit` cases.
- Extend the repack validator (`validate_mlx_repack_full.py`) → an *arbitrary* (non-designed)
  input sequence, exercising `repack()`.
- Tolerances = existing: logit/logprob `max|Δ|` within the current design tolerance; side-chain
  RMSD ~1e-6 Å.

**Tier 2 — Swift package tests (`MPNNKit/Tests/MPNNKitTests/MPNNModelTests.swift`).**

- **Regression guard (the gate):** greedy + `seed=0` `run()` is byte-identical after the rewrap.
- `design(fixedPositions:)` holds fixed positions exactly at native; only the complement changes.
- `bias` moves the argmax toward up-weighted AAs; `omit` never samples forbidden AAs.
- `score`: modes match the oracle within tol; `.unconditional` / `.leaveOneOut` are
  seed-independent (two seeds → identical); `currentAALogProb[i] == logProbs[i][sequence[i]]`.
- `repack(design.indices).pdb == run().pdb` — proves `run()` == `design()` + `repack()`.
- Each `MPNNInputError` fires on its bad input.
- Fixtures reuse `export_sim_fixtures.py` / `sim_test/`, plus a tiny multi-chain case.

**Tier 3 — perf reality-check (issue's open question).** Time `score` and partial `design` at
L ≈ 68 / 500 / 1500 / 2120, recording the encode-vs-decode split. This makes explicit the **O(L²)
truth: `fixedPositions` trims only decode work; the featurizer + encoder run over the whole protein
regardless of selection size** — so the <3 s/<500 aa and <10 s/<1500 aa budgets track total L, not
selection size.

## 9. Non-goals & risks (Phase 1)

- Protein-only (ligand context zeroed) — unchanged.
- All-atom repack coordinates trusted wholesale — no engine-side clash relaxation/sculpt.
- No partial-encode optimization — the whole-protein encode is accepted (documented, measured in
  Tier 3).
- iOS memory: MLX peak ~3 GB at ~2k residues is a real jetsam risk; **not** solved here (no
  chunking) — flagged as a Phase-2/host constraint (L cap), tied to RayMol's OOM history.
- `.leaveOneOut` costs more than `.conditional` (denser attention) — acceptable, measured.
- `.conditional` scores depend on decode order — documented, pinned via fixed `seed`.
- No `.mpnnpack` / weights / `Package.swift` change.

## 10. File map (in `proteinmpnn-ios`)

- `MPNNKit/Sources/MPNNKit/MPNNModel.swift` — public API (`alphabet`, `ScoreMode`, the
  result/options structs, `score`/`design`/`repack`, `MPNNInputError`); `run()` rewrapped;
  `Result` optionally gains `logits`; the `[Int] → S` helper.
- `MPNNKit/Sources/MPNNKit/DesignModel.swift` — nil-guarded `bias`/`omit` addend in
  `decodeSequence`; two sibling decodes (`decodeUnconditional`, `decodeLeaveOneOut`).
- `MPNNKit/Sources/MPNNKit/RepackLoop.swift` — unchanged.
- `MPNNKit/Tests/MPNNKitTests/MPNNModelTests.swift` — the Tier-2 tests.
- `port/capture_score.py`, `port/validate_mlx_score.py`; extend `port/validate_mlx_design.py` and
  the repack validator.
- `MPNNKit/README.md` — document the primitives + threading/determinism notes.

## 11. Acceptance criteria (Phase 1)

- [ ] `score(mode:)` returns `[L,21]` logprobs matching the reference oracle within tolerance for
      all three modes; `.unconditional`/`.leaveOneOut` are seed-independent.
- [ ] `currentAALogProb` equals the per-residue current-AA column of `logProbs`.
- [ ] `design(fixedPositions:nativeSequence:)` holds the fixed positions exactly and redesigns only
      the complement; `bias`/`omit` demonstrably steer/constrain the output.
- [ ] `repack(residues, sequence)` returns a valid PDB + per-atom confidence; matches `run()`'s
      repack for the designed sequence.
- [ ] `run()` output is byte-identical (greedy + `seed=0`) after the rewrap.
- [ ] No MLX type appears in any public signature.
- [x] Perf numbers recorded across L (`docs/perf/2026-07-21-mpnnkit-perf-lsweep.md`); encode measured
      **~linear** (log-log slope ≈0.95–1.0) — the O(L²) all-pairs step is GPU-parallel and negligible
      to L=2120. See §13.

## 12. Open questions carried into Phase 2 (RayMol), not blocking Phase 1

- Which `ScoreMode` gives the most useful heatmap in practice (default `.conditional` is a starting
  point; validated against real cases in Phase 2).
- How RayMol maps a named selection ↔ the dense `0..<L` residue index array (insertion codes, gaps,
  multi-chain) — reuse `appkit_sequence_panel._get_sequences()` ordering.
- The undo model given "new object per result", the confidence-legend UI, and the hover-vs-tap
  trigger on touch platforms (iPad/iPhone have no pointer hover today).

## 13. As-built corrections (post-implementation, 2026-07-21)

Recorded after the branch `design/mpnnkit-interactive-design-api` completed (commits
`75c4a40..a6f0220`, full suite 21/21, whole-branch review = merge-ready). These supersede the
earlier text where noted.

- **Scoring is deterministic; `seed` is inert (corrects §4/§6/§8/§9).** Implementation showed
  `.leaveOneOut` is *not* truly order-independent under the multi-layer decoder (~0.49 logprob
  variance across base orders), and Swift's MLX RNG cannot reproduce the PyTorch oracle's random
  order. Resolved by pinning an **RNG-free canonical decode order** on BOTH the oracle and Swift:
  `.conditional` uses the identity order `[0..L-1]` (position *i* conditioned on `0..i-1`);
  `.leaveOneOut` uses `[all others in identity order, idx last]` per position; `.unconditional` is
  genuinely order-free (strict zeroed backward mask). `score()` output does not depend on `seed`
  (the parameter is retained for API symmetry only). Parity vs the oracle is ~1e-5 (Linf).
- **`biasShapeMismatch` stays `(expected: Int, got: Int)` (corrects §7).** The tuple form shown in
  §7 cannot compile under `MPNNInputError: Equatable` (Swift does not auto-synthesize Equatable
  over tuple associated values). Outer-count vs inner-row (≠21) mismatches are distinguished by the
  thrown payload instead. `omit` length is also validated to equal `L` (parity with `bias`).
- **Oracle implementation (refines §8).** The loaded LigandMPNN class exposes only
  `score`/`single_aa_score`, so `port/capture_score.py` reuses the model's `encode()` and
  reimplements the three decodes mirroring the ProteinMPNN reference (`conditional` verified
  byte-identical, Δ=0, to `score(use_sequence=True)`). The planned `port/validate_mlx_score.py`
  MLX-mirror and the design/repack validator extensions were **not** added — the Swift package
  tests validate directly against the exported `.safetensors` oracle (equivalent trust, simpler).
- **Tests span multiple files (refines §8/§10):** `APIInputTests`, `DesignAPITests`,
  `RepackAPITests`, `RunRewrapTests`, `ScoreAPITests`, `TestFixtures` — not a single
  `MPNNModelTests.swift`. Repack parity uses the **designed** sequence (`design_top1`), matching how
  the repack oracle was generated.
- **`Result.logits` was not added** — `Result` is unchanged (spec §4 said "optionally").
- **Tier-3 perf L-sweep — DONE (acceptance §11 closed).** Ran a 9-point sweep (L=68…2120) under
  `-c release`; results in `docs/perf/2026-07-21-mpnnkit-perf-lsweep.md`. Key corrections to the
  spec's cost model:
  - **Encode is ~linear, not O(L²)** over the realistic range (slope ≈0.95–1.0, ~0.06 ms/residue) —
    the `topK=32` cap makes encoder work O(L·K); the O(L²) all-pairs step is GPU-parallel and
    negligible to L=2120. The O(L²) is a latent asymptote, not a practical cost here.
  - **`fixedPositions` is NOT a decode-time optimisation** — `decodeSequence` iterates all L
    positions regardless (fixed positions are *assigned*, not skipped), so a 10% redesign is only
    ~4–18% faster than a full one. Wall-clock tracks **total L, not selection size** — even more
    strongly than §9 stated. Phase-2 UX must reflect this.
  - Budgets comfortably met (design ≈0.45 s @425 aa, ≈1.5 s @1590 aa). Peak GPU memory ~linear,
    **~2.4 GB @ L=2120** — confirms the iOS jetsam risk / L-cap need from §9.
