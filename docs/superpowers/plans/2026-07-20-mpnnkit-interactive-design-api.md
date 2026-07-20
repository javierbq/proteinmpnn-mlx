# MPNNKit Interactive-Design API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add composable, MLX-free public primitives to the `MPNNKit` Swift package — `score(mode:)` (per-residue `[L,21]` log-probabilities), `design()` with fixed positions + per-position bias/omit, and standalone `repack()` — and re-express `run()` over them without changing its output, so RayMol #217's interactive design tool can be built on top in a later phase.

**Architecture:** Every capability already exists inside the model; this is mostly *exposure*. New public methods build the MLX inputs from `[Residue]`, call the existing internal `featuresDesignE → encodeDesign → decodeSequence`/`repackFull` pipeline, and convert results to Swift value types at the boundary. The only edits to the design hot path are (a) a nil-guarded additive logit-bias tensor inside `decodeSequence` and (b) two new sibling decode functions for the order-independent score modes. Correctness is enforced by parity tests against the existing `.safetensors` oracles and a new score oracle captured from the PyTorch reference.

**Tech Stack:** Swift 6 / SPM, mlx-swift 0.31.6 (Metal), XCTest. Python 3.12 (`../.venv/bin/python`), PyTorch LigandMPNN/ProteinMPNN references, `mlx.core` for oracle export.

## Global Constraints

- **MLX-free public boundary.** No MLX type may appear in any public signature — coords in as `SIMD3<Float>`; out as `String` / `[[Float]]` / `[Int]`. (`MPNNModel.swift:1-3, 52`.)
- **`run()` stays bit-stable.** Greedy + `seed = 0` must produce a byte-identical `sequence` and `pdb` after the rewrap. Guarded by a characterization test (Task 5).
- **`.mpnnpack`, `Package.swift`, and the weights do not change.** No new SPM dependency.
- **Alphabet is fixed:** `"ACDEFGHIKLMNPQRSTVWYX"`, 21 chars, index 20 = `X` (`DesignModel.swift:8`). All `[L,21]` outputs use this column order. `X` is never sampled by design (sampler slices `[0..<20]`, `DesignModel.swift:151,153`), but the `X` column is present in every returned `logits`/`logProbs`.
- **chain_mask convention:** `1.0` = designable/variable, `0.0` = fixed/native. `cm = mask * chainMask` (`DesignModel.swift:129`); `St = St*cmT + SnT*(1-cmT)` forces the native token where `chainMask == 0` (`DesignModel.swift:156-158`).
- **Threading:** all methods are synchronous and touch MLX/Metal only; the caller runs them off the main thread. No `cmd`/PyMOL calls exist here.
- **Determinism:** `seed` seeds `MLXRandom` (`MPNNModel.swift:60`); `temperature <= 0` ⇒ greedy. `.unconditional`/`.leaveOneOut` scoring is order-independent (seed ignored); `.conditional` scoring depends on the seeded decode order.
- **Test runner:** `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test` (macOS host, real Metal). No `-skipPackagePluginValidation` needed for `swift test`. Tests locate fixtures by walking up 4 dirs from `#filePath` to the repo root and reading `dist/MPNN.mpnnpack` + `app/MPNNBench/Resources/app_assets/{inputs,oracles}/*.safetensors`; guard every test with `try XCTSkipUnless(FileManager.default.fileExists(...))`.
- **Parity tolerances (reuse verbatim):** logits/log-probs `max|Δ| < 1e-3`; top-1 match `== 100%`; repack side-chain RMSD `< 1e-2 Å`, `max|Δ| < 5e-2 Å`.

**File structure (all under `/Users/jcastellanos/repos/proteinmpnn-ios`):**

- `MPNNKit/Sources/MPNNKit/MPNNModel.swift` — public API: `alphabet`, `MPNNInputError`, `ScoreMode`, `ScoreResult`, `DesignOptions`, `DesignResult`, `RepackResult`, `score`/`design`/`repack`, internal helpers (`modelInputs`, `buildLogitBias`, `toRows`, `repackPDBCore`); `run()` rewrapped.
- `MPNNKit/Sources/MPNNKit/DesignModel.swift` — nil-guarded `logitBias` param in `decodeSequence`; new `scoreUnconditional` / `scoreLeaveOneOut`.
- `MPNNKit/Tests/MPNNKitTests/APIInputTests.swift` — scaffolding/validation tests (Task 1).
- `MPNNKit/Tests/MPNNKitTests/DesignAPITests.swift` — decode regression + design behavioral tests (Tasks 2–3).
- `MPNNKit/Tests/MPNNKitTests/RepackAPITests.swift` — repack parity (Task 4).
- `MPNNKit/Tests/MPNNKitTests/RunRewrapTests.swift` — `run()` bit-stability (Task 5).
- `MPNNKit/Tests/MPNNKitTests/ScoreAPITests.swift` — score parity + order-independence (Task 7).
- `MPNNKit/Tests/MPNNKitTests/TestFixtures.swift` — shared `repoRoot`, `loadResidues`, `loadOracle` helpers used by the above.
- `port/capture_score.py` — new PyTorch→`.safetensors` score oracle generator (Task 6).
- `MPNNKit/README.md` — document the new primitives (Task 8).

**Task dependency map:** 1 → {2 → 3, 4}; {3, 4} → 5; 1 → {6 → 7}; 8 last. Tasks 6–7 (scoring) are independent of Tasks 2–5 (editing) and may be executed/reviewed in parallel by a different worker.

---

### Task 1: API scaffolding — `alphabet`, input builder, errors, bias/row helpers

**Files:**
- Modify: `MPNNKit/Sources/MPNNKit/MPNNModel.swift` (add members to `struct MPNNModel`)
- Create: `MPNNKit/Tests/MPNNKitTests/TestFixtures.swift`
- Create: `MPNNKit/Tests/MPNNKitTests/APIInputTests.swift`

**Interfaces:**
- Produces:
  - `public static var alphabet: [Character]` (== `ACDEFGHIKLMNPQRSTVWYX`)
  - `public enum MPNNInputError: Error, Equatable`
  - internal `func modelInputs(_:) -> (X: MLXArray, mask: MLXArray, Ridx: MLXArray, chainLabels: MLXArray)`
  - internal `func buildLogitBias(L:bias:omit:) -> MLXArray?`
  - internal `func toRows(_:rows:cols:) -> [[Float]]`
  - test helpers `repoRoot`, `packURL`, `loadResidues(_ id:) -> [MPNNModel.Residue]?`

- [ ] **Step 1: Write the shared test-fixture helper** (`TestFixtures.swift`)

```swift
import XCTest
import MLX
@testable import MPNNKit

/// Repo root = up 4 from MPNNKit/Tests/MPNNKitTests/<thisfile>.swift.
func mpnnRepoRoot(_ file: StaticString = #filePath) -> URL {
    URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
}

func mpnnPackURL() -> URL { mpnnRepoRoot().appendingPathComponent("dist/MPNN.mpnnpack") }

func mpnnInputURL(_ id: String) -> URL {
    mpnnRepoRoot().appendingPathComponent("app/MPNNBench/Resources/app_assets/inputs/\(id).safetensors")
}
func mpnnOracleURL(_ id: String, _ kind: String) -> URL {
    mpnnRepoRoot().appendingPathComponent("app/MPNNBench/Resources/app_assets/oracles/\(id)_\(kind).safetensors")
}

/// Build [Residue] from an input fixture (keys X [1,L,4,3] f32, R_idx i32, chain_labels i32).
func loadResidues(_ id: String) throws -> [MPNNModel.Residue] {
    let a = try loadArrays(url: mpnnInputURL(id))
    let X = a["X"]!.asType(.float32).asArray(Float.self)
    let ridx = a["R_idx"]!.asType(.int32).asArray(Int32.self)
    let chain = a["chain_labels"]!.asType(.int32).asArray(Int32.self)
    let L = ridx.count
    return (0 ..< L).map { i in
        func v(_ atom: Int) -> SIMD3<Float> { let b = (i * 4 + atom) * 3; return SIMD3(X[b], X[b + 1], X[b + 2]) }
        return .init(n: v(0), ca: v(1), c: v(2), o: v(3), chain: Int(chain[i]), resSeq: Int(ridx[i]))
    }
}

/// Native sequence (alphabet indices) from an input fixture (key S_native i32), or nil.
func loadNative(_ id: String) throws -> [Int]? {
    let a = try loadArrays(url: mpnnInputURL(id))
    guard let s = a["S_native"] else { return nil }
    return s.asType(.int32).asArray(Int32.self).map { Int($0) }
}

/// Skip a test unless the pack + a given input fixture are present.
func skipUnlessAssets(_ id: String, _ msg: String = "pack/fixture not present") throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnPackURL().path)
        && FileManager.default.fileExists(atPath: mpnnInputURL(id).path), msg)
}
```

- [ ] **Step 2: Write the failing scaffolding test** (`APIInputTests.swift`)

```swift
import XCTest
import MLX
@testable import MPNNKit

final class APIInputTests: XCTestCase {
    func testAlphabet() {
        XCTAssertEqual(String(MPNNModel.alphabet), "ACDEFGHIKLMNPQRSTVWYX")
        XCTAssertEqual(MPNNModel.alphabet.count, 21)
        XCTAssertEqual(MPNNModel.alphabet[20], "X")
    }

    func testModelInputsShapes() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let inp = model.modelInputs(residues)
        XCTAssertEqual(inp.X.shape, [1, residues.count, 4, 3])
        XCTAssertEqual(inp.mask.shape, [1, residues.count])
        XCTAssertEqual(inp.Ridx.shape, [1, residues.count])
        XCTAssertEqual(inp.chainLabels.shape, [1, residues.count])
    }

    func testBuildLogitBiasNilWhenBothNil() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        XCTAssertNil(model.buildLogitBias(L: 10, bias: nil, omit: nil))
    }

    func testBuildLogitBiasShapeAndOmit() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let b = model.buildLogitBias(L: 3, bias: nil, omit: [[], [5], []])
        XCTAssertEqual(b?.shape, [1, 3, 21])
        let flat = b![0].asType(.float32).asArray(Float.self)
        XCTAssertLessThan(flat[1 * 21 + 5], -1e8)   // omitted AA gets a large negative
        XCTAssertEqual(flat[0], 0)                    // untouched entry stays 0
    }

    func testToRows() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let a = MLXArray([1, 2, 3, 4, 5, 6].map { Float($0) }, [2, 3])
        let rows = model.toRows(a, rows: 2, cols: 3)
        XCTAssertEqual(rows, [[1, 2, 3], [4, 5, 6]])
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter APIInputTests`
Expected: FAIL to compile — `MPNNModel.alphabet`, `modelInputs`, `buildLogitBias`, `toRows` don't exist yet.

- [ ] **Step 4: Add the members to `MPNNModel`** (`MPNNModel.swift`, inside `struct MPNNModel`, e.g. after the `Result` struct)

```swift
/// Index → 1-letter AA (index 20 = 'X'). Column order of every [L,21] output.
public static var alphabet: [Character] { ALPHABET }   // ALPHABET is DesignModel.swift:8

public enum MPNNInputError: Error, Equatable {
    case emptyResidues
    case sequenceLengthMismatch(expected: Int, got: Int)
    case sequenceRequired(ScoreMode)
    case nativeSequenceRequired
    case indexOutOfRange(Int)
    case biasShapeMismatch(expected: Int, got: Int)
}

/// Build the model input tensors from residues (extracted verbatim from run(), MPNNModel.swift:62-70).
func modelInputs(_ residues: [Residue]) -> (X: MLXArray, mask: MLXArray, Ridx: MLXArray, chainLabels: MLXArray) {
    let L = residues.count
    var flat = [Float](); flat.reserveCapacity(L * 12)
    for r in residues { for v in [r.n, r.ca, r.c, r.o] { flat.append(v.x); flat.append(v.y); flat.append(v.z) } }
    let X = MLXArray(flat, [1, L, 4, 3])
    let mask = MLXArray.ones([1, L])
    let Ridx = MLXArray(residues.map { Float($0.resSeq) }, [1, L])
    let chainLabels = MLXArray(residues.map { Float($0.chain) }, [1, L])
    return (X, mask, Ridx, chainLabels)
}

/// [1,L,21] additive logit bias from an optional L×21 bias and per-position omit sets. nil if both nil.
func buildLogitBias(L: Int, bias: [[Float]]?, omit: [Set<Int>]?) -> MLXArray? {
    guard bias != nil || omit != nil else { return nil }
    var flat = [Float](repeating: 0, count: L * 21)
    if let bias = bias { for i in 0 ..< L { for a in 0 ..< 21 { flat[i * 21 + a] += bias[i][a] } } }
    if let omit = omit { for i in 0 ..< Swift.min(L, omit.count) { for a in omit[i] where a >= 0 && a < 21 { flat[i * 21 + a] += -1e9 } } }
    return MLXArray(flat, [1, L, 21])
}

/// Row-major [rows,cols] MLXArray → [[Float]].
func toRows(_ a: MLXArray, rows: Int, cols: Int) -> [[Float]] {
    let flat = a.asType(.float32).asArray(Float.self)
    return (0 ..< rows).map { i in Array(flat[(i * cols) ..< ((i + 1) * cols)]) }
}
```

Note: `ScoreMode` is referenced by `MPNNInputError`; it is added in Task 7. Until then, temporarily place `public enum ScoreMode: Equatable { case conditional, unconditional, leaveOneOut }` above `MPNNInputError` (Task 7 keeps this exact definition — do not duplicate it).

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter APIInputTests`
Expected: PASS (5 tests; the asset-gated ones run because `dist/MPNN.mpnnpack` + `6MRR` fixture are present in this checkout).

- [ ] **Step 6: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/Sources/MPNNKit/MPNNModel.swift MPNNKit/Tests/MPNNKitTests/TestFixtures.swift MPNNKit/Tests/MPNNKitTests/APIInputTests.swift
git commit -m "feat(mpnnkit): API scaffolding — alphabet, modelInputs, buildLogitBias, MPNNInputError"
```

---

### Task 2: `decodeSequence` gains a nil-guarded `logitBias` param (design-path regression guard)

**Files:**
- Modify: `MPNNKit/Sources/MPNNKit/DesignModel.swift:107-109` (signature) and `:146-147` (addend)
- Create: `MPNNKit/Tests/MPNNKitTests/DesignAPITests.swift`

**Interfaces:**
- Consumes: `decodeSequence(...)` (`DesignModel.swift:107`), design oracles `app/MPNNBench/Resources/app_assets/oracles/{id}_design.safetensors` (keys `decoding_order` i32/i64, `design_top1` i32, `design_logits` f32).
- Produces: `decodeSequence(..., logitBias: MLXArray? = nil)` — when `logitBias == nil` the decode is byte-identical to today.

- [ ] **Step 1: Write the failing regression test** (`DesignAPITests.swift`)

This mirrors the app-level parity pattern (calls the internal decode directly with the oracle's decoding order) — proving the new param did not perturb the design path.

```swift
import XCTest
import MLX
@testable import MPNNKit

final class DesignAPITests: XCTestCase {
    // Load a design oracle: decoding_order [1,L] i32, design_top1 [1,L] i32, design_logits [1,L,21] f32.
    private func designOracle(_ id: String) throws -> (order: MLXArray, top1: MLXArray, logits: MLXArray) {
        let a = try loadArrays(url: mpnnOracleURL(id, "design"))
        return (a["decoding_order"]!.asType(.int32), a["design_top1"]!.asType(.int32), a["design_logits"]!.asType(.float32))
    }

    func testDecodeLogitBiasNilMatchesDesignOracle() throws {
        for id in ["6MRR", "5L33"] {
            try skipUnlessAssets(id)
            try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "design").path), "design oracle missing")
            let model = try MPNNModel(packDirectory: mpnnPackURL())
            let residues = try loadResidues(id)
            let L = residues.count
            let inp = model.modelInputs(residues)
            let ora = try designOracle(id)
            let (E, eIdx) = featuresDesignE(model.designWeights, inp.X, inp.mask, inp.Ridx, inp.chainLabels, topK: 32)
            let (hV, hE) = encodeDesign(model.designWeights, E, eIdx, inp.mask)
            let Snative = MLXArray.zeros([1, L]).asType(.int32)
            let chainMask = MLXArray.ones([1, L])
            // logitBias defaults to nil → must reproduce the oracle exactly.
            let (S, logits, _) = decodeSequence(model.designWeights, hV, hE, eIdx, Snative, inp.mask, chainMask,
                                                ora.order, mode: .greedy)
            let top1 = mean((S .== ora.top1).asType(.float32)).item(Float.self)
            let dLogits = max(abs(logits - ora.logits)).item(Float.self)
            XCTAssertEqual(top1, 1.0, "\(id): design top-1 must match oracle exactly")
            XCTAssertLessThan(dLogits, 1e-3, "\(id): logits within 1e-3 of oracle")
        }
    }
}
```

This test needs `model.designWeights` to be reachable from tests. The stored property is `private let designW` (`MPNNModel.swift:41`). Add an internal accessor in this step's implementation.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter DesignAPITests/testDecodeLogitBiasNilMatchesDesignOracle`
Expected: FAIL to compile — `model.designWeights` is not accessible.

- [ ] **Step 3: Expose the weights + add the `logitBias` param**

In `MPNNModel.swift`, add an internal accessor (leave `designW` private):

```swift
var designWeights: Weights { designW }
var packerWeights: Weights { packerW }
```

In `DesignModel.swift`, change the signature (`:107-109`) to add the parameter:

```swift
func decodeSequence(_ w: Weights, _ hV: MLXArray, _ hE: MLXArray, _ eIdx: MLXArray,
                    _ Snative: MLXArray, _ mask: MLXArray, _ chainMask: MLXArray,
                    _ order: MLXArray, mode: SampleMode, nDec: Int = 3,
                    logitBias: MLXArray? = nil)
    -> (S: MLXArray, logits: MLXArray, logProbs: MLXArray) {
```

Inside the per-position loop, replace the single logits line (`DesignModel.swift:146`) with a nil-guarded addend:

```swift
        let logits0 = linear(w, "W_out", hVtF)                            // [B,21]
        let logits = logitBias == nil ? logits0
                     : logits0 + logitBias![0..., t ..< (t + 1)].reshaped([B, 21])
        let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
```

(Everything downstream — `argMax(logits[0..., 0 ..< 20], …)`, `logitsOut`, `logpOut` — is unchanged and now sees the biased `logits`.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter DesignAPITests/testDecodeLogitBiasNilMatchesDesignOracle`
Expected: PASS for both 6MRR and 5L33 (top1 == 1.0, dLogits < 1e-3).

- [ ] **Step 5: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/Sources/MPNNKit/DesignModel.swift MPNNKit/Sources/MPNNKit/MPNNModel.swift MPNNKit/Tests/MPNNKitTests/DesignAPITests.swift
git commit -m "feat(mpnnkit): nil-guarded logitBias in decodeSequence (design-path regression guarded)"
```

---

### Task 3: `design()` — fixed positions, per-position bias/omit, returned logits

**Files:**
- Modify: `MPNNKit/Sources/MPNNKit/MPNNModel.swift` (add `DesignOptions`, `DesignResult`, `design(_:options:)`)
- Modify: `MPNNKit/Tests/MPNNKitTests/DesignAPITests.swift` (add behavioral tests)

**Interfaces:**
- Consumes: `modelInputs`, `buildLogitBias`, `toRows` (Task 1); `featuresDesignE`, `encodeDesign`, `decodeSequence(..., logitBias:)` (Task 2); `designWeights` (Task 2).
- Produces:
  - `public struct DesignOptions { temperature; seed; fixedPositions: Set<Int>; nativeSequence: [Int]?; bias: [[Float]]?; omit: [Set<Int>]? }`
  - `public struct DesignResult { sequence: String; indices: [Int]; logits: [[Float]] }`
  - `public func design(_ residues: [Residue], options: DesignOptions = DesignOptions()) throws -> DesignResult`

- [ ] **Step 1: Write the failing behavioral tests** (append to `DesignAPITests.swift`)

```swift
    func testDesignHoldsFixedPositions() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let L = residues.count
        // A native sequence to hold fixed at a few positions (use all-Ala for a deterministic check).
        let native = [Int](repeating: 0, count: L)   // 0 == 'A'
        var opts = MPNNModel.DesignOptions()
        opts.temperature = 0; opts.seed = 0
        opts.fixedPositions = [0, 5, 10]
        opts.nativeSequence = native
        let r = try model.design(residues, options: opts)
        XCTAssertEqual(r.indices.count, L)
        for p in [0, 5, 10] { XCTAssertEqual(r.indices[p], 0, "fixed position \(p) must stay native (A)") }
    }

    func testDesignBiasForcesArgmax() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let L = residues.count
        let trp = 18   // index of 'W' in ACDEFGHIKLMNPQRSTVWYX
        var bias = Array(repeating: [Float](repeating: 0, count: 21), count: L)
        for i in 0 ..< L { bias[i][trp] = 1000 }        // overwhelming bias toward W
        var opts = MPNNModel.DesignOptions(); opts.temperature = 0; opts.seed = 0; opts.bias = bias
        let r = try model.design(residues, options: opts)
        XCTAssertTrue(r.indices.allSatisfy { $0 == trp }, "huge +bias on W must make every position W")
    }

    func testDesignOmitExcludesAA() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let L = residues.count
        var opts0 = MPNNModel.DesignOptions(); opts0.temperature = 0; opts0.seed = 0
        let free = try model.design(residues, options: opts0)
        // Pick an AA the free design actually used, then omit it everywhere.
        let used = Set(free.indices)
        let victim = used.first { $0 != 20 }!
        var opts = MPNNModel.DesignOptions(); opts.temperature = 0; opts.seed = 0
        opts.omit = Array(repeating: [victim], count: L)
        let r = try model.design(residues, options: opts)
        XCTAssertFalse(r.indices.contains(victim), "omitted AA must not appear")
    }

    func testDesignRejectsBadInput() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        var opts = MPNNModel.DesignOptions(); opts.fixedPositions = [0]   // nativeSequence missing
        XCTAssertThrowsError(try model.design(residues, options: opts)) { err in
            XCTAssertEqual(err as? MPNNModel.MPNNInputError, .nativeSequenceRequired)
        }
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter DesignAPITests`
Expected: FAIL to compile — `MPNNModel.DesignOptions` / `design` don't exist.

- [ ] **Step 3: Implement `design()`** (`MPNNModel.swift`)

```swift
public struct DesignOptions {
    public var temperature: Float = 0.1
    public var seed: UInt64? = nil
    public var fixedPositions: Set<Int> = []
    public var nativeSequence: [Int]? = nil     // alphabet indices, length L; required iff fixedPositions non-empty
    public var bias: [[Float]]? = nil           // L × 21 additive logit bias
    public var omit: [Set<Int>]? = nil          // per-position disallowed alphabet indices
    public init() {}
}

public struct DesignResult {
    public let sequence: String
    public let indices: [Int]
    public let logits: [[Float]]
}

public func design(_ residues: [Residue], options: DesignOptions = DesignOptions()) throws -> DesignResult {
    guard !residues.isEmpty else { throw MPNNInputError.emptyResidues }
    let L = residues.count
    if !options.fixedPositions.isEmpty {
        guard options.nativeSequence != nil else { throw MPNNInputError.nativeSequenceRequired }
        for p in options.fixedPositions where p < 0 || p >= L { throw MPNNInputError.indexOutOfRange(p) }
    }
    if let nat = options.nativeSequence {
        guard nat.count == L else { throw MPNNInputError.sequenceLengthMismatch(expected: L, got: nat.count) }
        for a in nat where a < 0 || a >= 21 { throw MPNNInputError.indexOutOfRange(a) }
    }
    if let b = options.bias { guard b.count == L && b.allSatisfy({ $0.count == 21 }) else {
        throw MPNNInputError.biasShapeMismatch(expected: L, got: b.count) } }

    if let s = options.seed { MLXRandom.seed(s) }
    let (X, mask, Ridx, chainLabels) = modelInputs(residues)
    var cmArr = [Float](repeating: 1, count: L)
    for p in options.fixedPositions { cmArr[p] = 0 }
    let chainMask = MLXArray(cmArr, [1, L])
    let natInts = options.nativeSequence ?? [Int](repeating: 0, count: L)
    let Snative = MLXArray(natInts.map { Int32($0) }, [1, L]).asType(.int32)
    let logitBias = buildLogitBias(L: L, bias: options.bias, omit: options.omit)

    let order = argSort(MLXRandom.normal([1, L]), axis: -1).asType(.int32)
    let (E, eIdx) = featuresDesignE(designW, X, mask, Ridx, chainLabels, topK: 32)
    let (hV, hE) = encodeDesign(designW, E, eIdx, mask)
    let mode: SampleMode = options.temperature <= 0 ? .greedy : .sample(temperature: options.temperature)
    let (S, logits, _) = decodeSequence(designW, hV, hE, eIdx, Snative, mask, chainMask, order,
                                        mode: mode, logitBias: logitBias)
    MLX.eval(S, logits)
    let idx = S[0].asType(.int32).asArray(Int32.self).map { Int($0) }
    return DesignResult(sequence: String(idx.map { Self.alphabet[$0] }),
                        indices: idx,
                        logits: toRows(logits[0], rows: L, cols: 21))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter DesignAPITests`
Expected: PASS (regression test from Task 2 + the four new behavioral tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/Sources/MPNNKit/MPNNModel.swift MPNNKit/Tests/MPNNKitTests/DesignAPITests.swift
git commit -m "feat(mpnnkit): design() with fixedPositions + per-position bias/omit + returned logits"
```

---

### Task 4: `repack()` — standalone side-chain repack + factored PDB core

**Files:**
- Modify: `MPNNKit/Sources/MPNNKit/MPNNModel.swift` (add `RepackResult`, `repack(_:sequence:)`, internal `repackPDBCore`)
- Create: `MPNNKit/Tests/MPNNKitTests/RepackAPITests.swift`

**Interfaces:**
- Consumes: `modelInputs` (Task 1); `packerWeights` (Task 2); internal `repackFull` (`RepackLoop.swift:42`), `PDBWriter.write` (`PDBWriter.swift:16`), `geom`, `names` (`MPNNModel` stored props).
- Produces:
  - `public struct RepackResult { pdb: String; atomConfidence: [[Float]] }`  (`atomConfidence` = L×14)
  - `public func repack(_ residues: [Residue], sequence: [Int]) throws -> RepackResult`
  - internal `func repackPDBCore(_ residues: [Residue], indices: [Int]) -> (pdb: String, bFactors: MLXArray)` (also used by `run()` in Task 5)

- [ ] **Step 1: Write the failing repack parity test** (`RepackAPITests.swift`)

```swift
import XCTest
import MLX
@testable import MPNNKit

final class RepackAPITests: XCTestCase {
    func testRepackMatchesOracleSideChains() throws {
        let id = "6MRR"
        try skipUnlessAssets(id)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "repack").path), "repack oracle missing")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues(id)
        let L = residues.count
        // Repack the NATIVE sequence (the repack oracle was generated for it).
        let native = try XCTUnwrap(loadNative(id))
        let out = try model.repack(residues, sequence: native)
        XCTAssertTrue(out.pdb.contains("ATOM  "))
        XCTAssertEqual(out.atomConfidence.count, L)
        XCTAssertEqual(out.atomConfidence.first?.count, 14)

        // Side-chain (atom14 idx 4..13) RMSD vs oracle atom14, masked.
        let ora = try loadArrays(url: mpnnOracleURL(id, "repack"))
        let refA14 = ora["atom14"]!.asType(.float32)              // [1,L,14,3]
        let refMask = ora["atom14_mask"]!.asType(.float32)        // [1,L,14]
        let mineA14 = model.repackAtom14(residues, indices: native)   // test-only helper (Step 3)
        let scSlice = 4 ..< 14
        let diff = (mineA14[0..., 0..., scSlice] - refA14[0..., 0..., scSlice])
        let m = refMask[0..., 0..., scSlice].expandedDimensions(axis: -1)
        let sq = (diff * diff * m)
        let rmsd = sqrt((sum(sq) / max(sum(m) * 3, MLXArray(1))).item(Float.self))
        XCTAssertLessThan(rmsd, 1e-2, "side-chain RMSD within 1e-2 Å of oracle")
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter RepackAPITests`
Expected: FAIL to compile — `repack`, `repackAtom14` don't exist.

- [ ] **Step 3: Implement `repack()` + the factored core** (`MPNNModel.swift`)

```swift
public struct RepackResult {
    public let pdb: String
    public let atomConfidence: [[Float]]   // L × 14 packer log-prob per atom slot
}

public func repack(_ residues: [Residue], sequence: [Int]) throws -> RepackResult {
    guard !residues.isEmpty else { throw MPNNInputError.emptyResidues }
    let L = residues.count
    guard sequence.count == L else { throw MPNNInputError.sequenceLengthMismatch(expected: L, got: sequence.count) }
    for a in sequence where a < 0 || a >= 21 { throw MPNNInputError.indexOutOfRange(a) }
    let core = repackCore(residues, indices: sequence)
    MLX.eval(core.atom14, core.atom14Mask, core.bFactors)
    let pdb = PDBWriter.write(backbone: core.X, atom14: core.atom14, atom14Mask: core.atom14Mask,
                              bFactors: core.bFactors, seqMPNN: sequence,
                              chainLabels: residues.map { $0.chain }, residueIdx: residues.map { $0.resSeq }, names: names)
    return RepackResult(pdb: pdb, atomConfidence: toRows(core.bFactors[0], rows: L, cols: 14))
}

/// Shared repack: returns the raw MLXArrays so both repack() and run()'s PDB path use one code path.
func repackCore(_ residues: [Residue], indices: [Int])
    -> (X: MLXArray, atom14: MLXArray, atom14Mask: MLXArray, bFactors: MLXArray) {
    let (X, mask, Ridx, chainLabels) = modelInputs(residues)
    let S = MLXArray(indices.map { Int32($0) }, [1, residues.count])
    let r = repackFull(packerW, geom, Xbb: X, S: S, mask: mask, Ridx: Ridx, chainLabels: chainLabels)
    return (X, r.atom14, r.atom14Mask, r.bFactors)
}

#if DEBUG
/// Test-only: expose the raw repacked atom14 for RMSD parity.
func repackAtom14(_ residues: [Residue], indices: [Int]) -> MLXArray {
    let c = repackCore(residues, indices: indices); MLX.eval(c.atom14); return c.atom14
}
#endif
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter RepackAPITests`
Expected: PASS (PDB has ATOM records; atomConfidence is L×14; side-chain RMSD < 1e-2 Å).

- [ ] **Step 5: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/Sources/MPNNKit/MPNNModel.swift MPNNKit/Tests/MPNNKitTests/RepackAPITests.swift
git commit -m "feat(mpnnkit): standalone repack() + factored repackCore (shared with run)"
```

---

### Task 5: Rewrap `run()` over `design()` + `repackCore()` (bit-stability guard)

**Files:**
- Modify: `MPNNKit/Sources/MPNNKit/MPNNModel.swift:57-97` (`run` body)
- Create: `MPNNKit/Tests/MPNNKitTests/RunRewrapTests.swift`

**Interfaces:**
- Consumes: `design(_:options:)` (Task 3), `repackCore` + `PDBWriter.write` (Task 4).
- Produces: unchanged `run(_:options:) -> Result` signature and output.

- [ ] **Step 1: Capture the current `run()` output as a golden (characterization)**

Before refactoring, record what `run()` produces so the refactor can be proven identical. Add this temporary test, run it once, and copy the two printed values into the assertions in Step 2.

```swift
// TEMPORARY — run once, copy printed values into RunRewrapTests, then delete this func.
func testCaptureRunGolden() throws {
    try skipUnlessAssets("6MRR")
    let model = try MPNNModel(packDirectory: mpnnPackURL())
    let residues = try loadResidues("6MRR")
    var opts = MPNNModel.Options(); opts.temperature = 0; opts.seed = 0
    let r = try model.run(residues, options: opts)
    print("GOLDEN_SEQ=\(r.sequence)")
    print("GOLDEN_PDB_SHA=\(r.pdb!.data(using: .utf8)!.sha256Hex())")   // add a tiny Data.sha256Hex helper, or print r.pdb!.count
}
```

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter RunRewrapTests/testCaptureRunGolden 2>&1 | grep GOLDEN_`
Expected: prints `GOLDEN_SEQ=...` and `GOLDEN_PDB_SHA=...` (or PDB length). Copy both.

If you prefer not to add a SHA helper, print `r.pdb!.count` and assert the count instead — the sequence equality is the primary bit-stability signal; the PDB is a deterministic function of the same `S` MLXArray so equal sequence + equal length is a strong guard.

- [ ] **Step 2: Write the failing bit-stability test** (`RunRewrapTests.swift`, replacing the temporary capture)

```swift
import XCTest
import MLX
@testable import MPNNKit

final class RunRewrapTests: XCTestCase {
    func testRunGreedySeed0IsBitStable() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        var opts = MPNNModel.Options(); opts.temperature = 0; opts.seed = 0
        let r = try model.run(residues, options: opts)
        let golden = "<PASTE GOLDEN_SEQ FROM STEP 1>"
        XCTAssertEqual(r.sequence, golden, "run() greedy+seed0 must be byte-identical after the rewrap")
        XCTAssertNotNil(r.pdb)
        XCTAssertTrue(r.pdb!.contains("ATOM  "))
    }
}
```

- [ ] **Step 3: Run to verify it PASSES against the current (pre-refactor) `run()`**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter RunRewrapTests/testRunGreedySeed0IsBitStable`
Expected: PASS (the golden was captured from the current `run()`). This is a characterization test — it must pass before AND after the refactor.

- [ ] **Step 4: Rewrap `run()`** (`MPNNModel.swift:57-97`)

```swift
public func run(_ residues: [Residue], options: Options = Options()) throws -> Result {
    precondition(!residues.isEmpty, "MPNNModel.run: empty residue list")
    var d = DesignOptions()
    d.temperature = options.temperature
    d.seed = options.seed
    let t0 = DispatchTime.now().uptimeNanoseconds
    let dr = try design(residues, options: d)
    let t1 = DispatchTime.now().uptimeNanoseconds

    var pdb: String? = nil
    var repackMs = 0.0
    if options.repack {
        let core = repackCore(residues, indices: dr.indices)
        MLX.eval(core.atom14, core.atom14Mask, core.bFactors)
        repackMs = Double(DispatchTime.now().uptimeNanoseconds - t1) / 1e6
        pdb = PDBWriter.write(backbone: core.X, atom14: core.atom14, atom14Mask: core.atom14Mask,
                              bFactors: core.bFactors, seqMPNN: dr.indices,
                              chainLabels: residues.map { $0.chain }, residueIdx: residues.map { $0.resSeq }, names: names)
    }
    return Result(sequence: dr.sequence, pdb: pdb,
                  designMs: Double(t1 - t0) / 1e6, repackMs: repackMs)
}
```

Bit-stability rationale (verify while implementing): `design()` seeds `MLXRandom` with `options.seed` then draws exactly one `MLXRandom.normal([1,L])` for `order` — the same RNG sequence the old `run()` used (`MPNNModel.swift:60,72`). `modelInputs`/validation draw no randoms; `repackFull` is RNG-free. `String→[Int]→MLXArray` for the sequence is exact for indices 0..20.

- [ ] **Step 5: Run to verify it still passes**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter RunRewrapTests`
Also run the original smoke test to be safe: `swift test --filter MPNNModelTests/testDesignRepackFrom6MRR`
Expected: PASS — identical sequence, PDB still valid.

- [ ] **Step 6: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/Sources/MPNNKit/MPNNModel.swift MPNNKit/Tests/MPNNKitTests/RunRewrapTests.swift
git commit -m "refactor(mpnnkit): run() rewrapped over design()+repackCore, bit-stability guarded"
```

---

### Task 6: Score parity oracle — `port/capture_score.py`

**Files:**
- Create: `port/capture_score.py`
- Output (git-tracked fixtures): `app/MPNNBench/Resources/app_assets/oracles/{id}_score.safetensors`

**Interfaces:**
- Consumes: the LigandMPNN reference at `/Users/jcastellanos/repos/LigandMPNN` (checkpoint `model_params/ligandmpnn_v_32_010_25.pt`), the existing `port/capture_design.py` feature-dict construction, `mlx.core` for safetensors export.
- Produces: per structure, `oracles/{id}_score.safetensors` with keys `native_seq` (i32 [L]), `logprobs_conditional` / `logprobs_unconditional` / `logprobs_leaveoneout` (f32 [L,21]), `decoding_order` (i32 [L], the seeded order used for `conditional`).

This task follows the repo's captured-fixture pattern (like `capture_design.py`). It produces the ground truth the Swift `score()` will match in Task 7. **These are the exact reference semantics** (verified against both reference repos):

- **conditional** = teacher-forced autoregressive over ONE fixed decode order. Reference: `ProteinMPNN.forward(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=True, decoding_order=<fixed>)` → `log_probs` (`/Users/jcastellanos/repos/ProteinMPNN/protein_mpnn_utils.py:1057`). Native `S` fed; each position sees residues earlier in the order.
- **unconditional** = structure only, no sequence. Reference: `ProteinMPNN.unconditional_probs(X, mask, residue_idx, chain_encoding_all)` — forces `order_mask_backward = 0` (`protein_mpnn_utils.py:1352,1370`). Order-independent. **This is the definition Swift `.unconditional` must match** (strict zeroed order — NOT LigandMPNN `score(use_sequence=False)`, which still uses a random order).
- **leaveOneOut** = each position conditioned on ALL other native residues. Reference: `ProteinMPNN.conditional_probs(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False)` (`protein_mpnn_utils.py:1292`; the per-idx `order_mask=zeros; order_mask[idx]=1` places idx LAST → sees all others).

**GOTCHA (must obey):** if you instead use LigandMPNN's `single_aa_score`, its `use_sequence` flag is wired **opposite** to its help text — leave-one-out requires `--single_aa_score 1 --use_sequence 0` (`/Users/jcastellanos/repos/LigandMPNN/model_utils.py:502-507`). To avoid this trap entirely, prefer `ProteinMPNN.conditional_probs(backbone_only=False)`, which is correctly named. The MPNNKit design weights are LigandMPNN's; load them into the reference `ProteinMPNN` architecture the same way `capture_design.py` does (it already imports the model + checkpoint).

- [ ] **Step 1: Write `capture_score.py`**

Model the structure on `port/capture_design.py` (which already loads the model, builds the feature dict via `parse_PDB`+`featurize`, and `np.savez`es stages). Reuse its `build_feature_dict(pdb_path)` (`capture_design.py:32`). For each mode, call the reference method with a **fixed** decode order (seed the reference `randn` / pass `decoding_order`), then convert to `mx.array` and `mx.save_safetensors`. Concrete skeleton:

```python
#!/usr/bin/env python3
"""Capture reference per-position log-probs (3 ScoreMode semantics) to .safetensors
oracle fixtures for the Swift MPNNKit score() parity tests. Mirrors capture_design.py."""
import os, sys, argparse
import numpy as np, torch
import mlx.core as mx

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "LigandMPNN")))
from model_utils import ProteinMPNN                       # noqa: E402
from data_utils import parse_PDB, featurize               # noqa: E402
sys.path.insert(0, HERE)
from capture_design import build_feature_dict             # reuse the exact featurizer

CKPT = os.path.abspath(os.path.join(HERE, "..", "..", "LigandMPNN", "model_params", "ligandmpnn_v_32_010_25.pt"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "MPNNBench", "Resources", "app_assets"))

def load_model():
    # copy the exact load used at the top of capture_design.py's main() (checkpoint, model_type,
    # use_atom_context=False, number_of_ligand_atoms=25, .eval())
    ...

def fixed_order(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    randn = torch.randn(1, L, generator=g)
    return torch.argsort(randn, dim=1)        # a deterministic decode order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True); ap.add_argument("--id", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model = load_model()
    fd, L = build_feature_dict(a.pdb)
    S = fd["S"]                                             # native sequence [1,L] i64
    order = fixed_order(L, a.seed)
    with torch.no_grad():
        cond   = model.forward(fd["X"], S, fd["mask"], fd["chain_mask"], fd["R_idx"],
                               fd["chain_labels"], randn=None,
                               use_input_decoding_order=True, decoding_order=order)   # [1,L,21] log_probs
        uncond = model.unconditional_probs(fd["X"], fd["mask"], fd["R_idx"], fd["chain_labels"])
        loo    = model.conditional_probs(fd["X"], S, fd["mask"], fd["chain_mask"], fd["R_idx"],
                                         fd["chain_labels"], randn=fixed_order_randn(L, a.seed),
                                         backbone_only=False)
    out = {
        "native_seq":            mx.array(S[0].cpu().numpy().astype(np.int32)),
        "logprobs_conditional":  mx.array(cond[0].cpu().numpy().astype(np.float32)),
        "logprobs_unconditional":mx.array(uncond[0].cpu().numpy().astype(np.float32)),
        "logprobs_leaveoneout":  mx.array(loo[0].cpu().numpy().astype(np.float32)),
        "decoding_order":        mx.array(order[0].cpu().numpy().astype(np.int32)),
    }
    os.makedirs(f"{ASSETS}/oracles", exist_ok=True)
    mx.save_safetensors(f"{ASSETS}/oracles/{a.id}_score.safetensors", out, metadata={"format": "mpnnbench"})
    print(f"[capture_score] {a.id} L={L} -> oracles/{a.id}_score.safetensors")

if __name__ == "__main__":
    main()
```

Fill the `...` sections against `capture_design.py` (the model load) and the exact argument names of `ProteinMPNN.forward`/`unconditional_probs`/`conditional_probs` (`protein_mpnn_utils.py:1057,1292,1352` — signatures listed in the plan header of Task 6). The reference `forward`/`conditional_probs` return `log_probs` directly (log-softmax). Keep the alphabet column order `ACDEFGHIKLMNPQRSTVWYX`.

- [ ] **Step 2: Generate the oracle fixtures**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios/port
../.venv/bin/python capture_score.py --pdb /Users/jcastellanos/repos/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb --id 6MRR
../.venv/bin/python capture_score.py --pdb /Users/jcastellanos/repos/ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb --id 5L33
```
Expected: two `oracles/{id}_score.safetensors` written; the printed `L` matches the input fixtures (6MRR L=68).

- [ ] **Step 3: Sanity-check the fixtures**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios/port
../.venv/bin/python -c "import mlx.core as mx; d=mx.load('../app/MPNNBench/Resources/app_assets/oracles/6MRR_score.safetensors'); print({k:v.shape for k,v in d.items()}); import math; print('rowsum~0', float(mx.max(mx.abs(mx.logsumexp(d['logprobs_unconditional'],axis=-1)))))"
```
Expected: shapes `logprobs_* = [L,21]`, `native_seq=[L]`, `decoding_order=[L]`; `logsumexp` over the AA axis ≈ 0 (valid log-probabilities).

- [ ] **Step 4: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add port/capture_score.py app/MPNNBench/Resources/app_assets/oracles/6MRR_score.safetensors app/MPNNBench/Resources/app_assets/oracles/5L33_score.safetensors
git commit -m "test(mpnnkit): score parity oracle — capture_score.py + 6MRR/5L33 score fixtures"
```

---

### Task 7: `score()` — three ScoreMode semantics + sibling decodes

**Files:**
- Modify: `MPNNKit/Sources/MPNNKit/MPNNModel.swift` (add `ScoreResult`, `score(...)`; `ScoreMode` was added in Task 1)
- Modify: `MPNNKit/Sources/MPNNKit/DesignModel.swift` (add `scoreUnconditional`, `scoreLeaveOneOut`)
- Create: `MPNNKit/Tests/MPNNKitTests/ScoreAPITests.swift`

**Interfaces:**
- Consumes: `modelInputs` (Task 1); `featuresDesignE`, `encodeDesign`, `decodeSequence` (existing); `designWeights` (Task 2); the score oracles (Task 6).
- Produces:
  - `public struct ScoreResult { logProbs: [[Float]]; currentAALogProb: [Float]? }`
  - `public func score(_ residues: [Residue], sequence: [Int]? = nil, mode: ScoreMode = .conditional, seed: UInt64? = 0) throws -> ScoreResult`
  - internal `func scoreUnconditional(...) -> MLXArray`, `func scoreLeaveOneOut(...) -> MLXArray`

- [ ] **Step 1: Write the failing parity + order-independence tests** (`ScoreAPITests.swift`)

```swift
import XCTest
import MLX
@testable import MPNNKit

final class ScoreAPITests: XCTestCase {
    private func scoreOracle(_ id: String, _ key: String) throws -> MLXArray {
        try loadArrays(url: mpnnOracleURL(id, "score"))[key]!.asType(.float32)
    }
    private func assertClose(_ mine: [[Float]], _ ref: MLXArray, _ tol: Float, _ msg: String) {
        let flat = mine.flatMap { $0 }
        let mineA = MLXArray(flat, ref.shape)
        XCTAssertLessThan(max(abs(mineA - ref)).item(Float.self), tol, msg)
    }

    func testConditionalParity() throws {
        for id in ["6MRR", "5L33"] {
            try skipUnlessAssets(id)
            try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "score").path), "score oracle missing")
            let model = try MPNNModel(packDirectory: mpnnPackURL())
            let residues = try loadResidues(id)
            let native = try XCTUnwrap(loadNative(id))
            let r = try model.score(residues, sequence: native, mode: .conditional, seed: 0)
            assertClose(r.logProbs, try scoreOracle(id, "logprobs_conditional"), 1e-3, "\(id) conditional")
            // currentAALogProb must equal the native-AA column of logProbs.
            let cur = try XCTUnwrap(r.currentAALogProb)
            for i in 0 ..< residues.count { XCTAssertEqual(cur[i], r.logProbs[i][native[i]], accuracy: 1e-6) }
        }
    }

    func testUnconditionalParityAndOrderIndependence() throws {
        let id = "6MRR"
        try skipUnlessAssets(id)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "score").path), "score oracle missing")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues(id)
        let a = try model.score(residues, mode: .unconditional, seed: 1)
        let b = try model.score(residues, mode: .unconditional, seed: 999)
        XCTAssertEqual(a.logProbs.flatMap { $0 }, b.logProbs.flatMap { $0 }, "unconditional must be seed-independent")
        assertClose(a.logProbs, try scoreOracle(id, "logprobs_unconditional"), 1e-3, "\(id) unconditional")
    }

    func testLeaveOneOutParity() throws {
        let id = "6MRR"
        try skipUnlessAssets(id)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "score").path), "score oracle missing")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues(id)
        let native = try XCTUnwrap(loadNative(id))
        let r = try model.score(residues, sequence: native, mode: .leaveOneOut, seed: 0)
        assertClose(r.logProbs, try scoreOracle(id, "logprobs_leaveoneout"), 1e-3, "\(id) leaveOneOut")
    }

    func testConditionalRequiresSequence() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        XCTAssertThrowsError(try model.score(residues, sequence: nil, mode: .conditional)) { err in
            XCTAssertEqual(err as? MPNNModel.MPNNInputError, .sequenceRequired(.conditional))
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter ScoreAPITests`
Expected: FAIL to compile — `score`, `ScoreResult`, `scoreUnconditional`, `scoreLeaveOneOut` don't exist.

- [ ] **Step 3: Add the two sibling decodes** (`DesignModel.swift`, after `decodeSequence`)

```swift
// UNCONDITIONAL: structure-only per-position marginals. Mirrors ProteinMPNN.unconditional_probs
// (order_mask_backward = 0 ⇒ decoder sees only the encoder forward features, never h_S).
// One pass; order-independent.
func scoreUnconditional(_ w: Weights, _ hV0: MLXArray, _ hE: MLXArray, _ eIdx: MLXArray,
                        _ mask: MLXArray, nDec: Int = 3) -> MLXArray {
    let B = hV0.dim(0), L = hV0.dim(1), C = hV0.dim(2)
    let hEXenc = catNeighborsNodesG(MLXArray.zeros([B, L, C]), hE, eIdx)   // zeros in place of h_S
    let hEXVenc = catNeighborsNodesG(hV0, hEXenc, eIdx)
    let hEXVfw = mask.reshaped([B, L, 1, 1]) * hEXVenc                      // mask_fw = mask, mask_bw = 0
    var hV = hV0
    for l in 0 ..< nDec { hV = decLayer(w, "decoder_layers.\(l)", hV, hEXVfw, maskV: mask) }
    let logits = linear(w, "W_out", hV)                                    // [B,L,21]
    MLX.eval(logits)
    return logits - logSumExp(logits, axis: -1, keepDims: true)
}

// LEAVE-ONE-OUT: p(AA_i | structure + all OTHER native residues). Mirrors
// ProteinMPNN.conditional_probs(backbone_only=false): for each i, place i LAST in the decode
// order so it attends to every other native residue in one pass. Order-independent; O(L) passes.
func scoreLeaveOneOut(_ w: Weights, _ hVenc: MLXArray, _ hE: MLXArray, _ eIdx: MLXArray,
                      _ Snative: MLXArray, _ mask: MLXArray, nDec: Int = 3) -> MLXArray {
    let B = hVenc.dim(0), L = hVenc.dim(1), C = hVenc.dim(2)
    let Ws = w["W_s.weight"]!
    let hS = take(Ws, Snative.asType(.int32), axis: 0)                     // [B,L,C]
    let hES = catNeighborsNodesG(hS, hE, eIdx)
    let hEXenc = catNeighborsNodesG(MLXArray.zeros([B, L, C]), hE, eIdx)
    let hEXVencoder = catNeighborsNodesG(hVenc, hEXenc, eIdx)
    var out = MLXArray.zeros([B, L, 21])
    for idx in 0 ..< L {
        var om = [Float](repeating: 0, count: L); om[idx] = 1              // idx decoded LAST
        let order = argSort(MLXArray(om, [1, L]) + 0.0001, axis: -1).asType(.int32)
        let maskAttend = gatherEdges(orderMaskBackward(order, L).expandedDimensions(axis: -1), eIdx)  // [B,L,K,1]
        let mask1D = mask.reshaped([B, L, 1, 1])
        let maskBw = mask1D * maskAttend
        let maskFw = mask1D * (1 - maskAttend)
        var hV = hVenc
        for l in 0 ..< nDec {
            let hESV = maskBw * catNeighborsNodesG(hV, hES, eIdx) + maskFw * hEXVencoder
            hV = decLayer(w, "decoder_layers.\(l)", hV, hESV, maskV: mask)
        }
        let logits = linear(w, "W_out", hV)
        let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
        out[0..., idx ..< (idx + 1)] = logp[0..., idx ..< (idx + 1)]
        MLX.eval(out)
    }
    return out
}
```

`orderMaskBackward` is `private` in `DesignModel.swift:95`. Change it to internal (drop `private`) so `scoreLeaveOneOut` can call it (same file, but keep it accessible if the siblings later move).

- [ ] **Step 4: Add `score()` + `ScoreResult`** (`MPNNModel.swift`)

```swift
public struct ScoreResult {
    public let logProbs: [[Float]]          // L × 21
    public let currentAALogProb: [Float]?   // per-residue log-prob of the current AA (iff `sequence` given)
}

public func score(_ residues: [Residue], sequence: [Int]? = nil,
                  mode: ScoreMode = .conditional, seed: UInt64? = 0) throws -> ScoreResult {
    guard !residues.isEmpty else { throw MPNNInputError.emptyResidues }
    let L = residues.count
    if let seq = sequence {
        guard seq.count == L else { throw MPNNInputError.sequenceLengthMismatch(expected: L, got: seq.count) }
        for a in seq where a < 0 || a >= 21 { throw MPNNInputError.indexOutOfRange(a) }
    }
    if (mode == .conditional || mode == .leaveOneOut) && sequence == nil { throw MPNNInputError.sequenceRequired(mode) }

    let (X, mask, Ridx, chainLabels) = modelInputs(residues)
    let (E, eIdx) = featuresDesignE(designW, X, mask, Ridx, chainLabels, topK: 32)
    let (hV, hE) = encodeDesign(designW, E, eIdx, mask)

    let logp: MLXArray
    switch mode {
    case .conditional:
        if let s = seed { MLXRandom.seed(s) }
        let order = argSort(MLXRandom.normal([1, L]), axis: -1).asType(.int32)
        let Snative = MLXArray(sequence!.map { Int32($0) }, [1, L]).asType(.int32)
        let (_, _, lp) = decodeSequence(designW, hV, hE, eIdx, Snative, mask, MLXArray.zeros([1, L]), order, mode: .greedy)
        logp = lp
    case .unconditional:
        logp = scoreUnconditional(designW, hV, hE, eIdx, mask)
    case .leaveOneOut:
        let Snative = MLXArray(sequence!.map { Int32($0) }, [1, L]).asType(.int32)
        logp = scoreLeaveOneOut(designW, hV, hE, eIdx, Snative, mask)
    }
    MLX.eval(logp)
    let rows = toRows(logp[0], rows: L, cols: 21)
    let cur = sequence.map { seq in (0 ..< L).map { rows[$0][seq[$0]] } }
    return ScoreResult(logProbs: rows, currentAALogProb: cur)
}
```

For `.conditional`, `chainMask = zeros` forces every position to `Snative` (the passed sequence) via `St = St*0 + SnT*1`, so `logp` is the teacher-forced conditional; `mode: .greedy` keeps it RNG-free (the sampled `St` is discarded).

- [ ] **Step 5: Run to verify it passes; iterate on the siblings until parity holds**

Run: `cd /Users/jcastellanos/repos/proteinmpnn-ios/MPNNKit && swift test --filter ScoreAPITests`
Expected: PASS. The parity tests (`< 1e-3` vs the Task 6 oracle) are the precise correctness spec for `scoreUnconditional`/`scoreLeaveOneOut` and for the `.conditional` decode config. If a mode fails parity, compare against the reference definition in Task 6 (mask construction, whether `h_S` is fed, which position is decoded last) and adjust the sibling — do not loosen the tolerance.

- [ ] **Step 6: Commit**

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/Sources/MPNNKit/MPNNModel.swift MPNNKit/Sources/MPNNKit/DesignModel.swift MPNNKit/Tests/MPNNKitTests/ScoreAPITests.swift
git commit -m "feat(mpnnkit): score() with conditional/unconditional/leaveOneOut modes + parity"
```

---

### Task 8: Document the new primitives

**Files:**
- Modify: `MPNNKit/README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a "Composable primitives" section to the README**

Document `score(_:sequence:mode:seed:)`, `design(_:options:)`, and `repack(_:sequence:)` with a short usage block each, mirroring the existing `run()` example (README.md:32-50). Cover:
- the `ScoreMode` semantics (conditional = current-AA log-prob given earlier-in-order residues; unconditional = structure only, order-independent; leaveOneOut = given all other native residues, order-independent, costlier);
- the alphabet index convention (`MPNNModel.alphabet`, index 20 = `X`);
- `DesignOptions.fixedPositions`/`nativeSequence`/`bias`/`omit` and that hard-assign = fix + set native, palette-restrict = bias/omit;
- determinism (`seed`; `.unconditional`/`.leaveOneOut` ignore it);
- threading (call off the main thread);
- a "Regenerating parity oracles" note pointing at `port/capture_score.py` and the LigandMPNN `single_aa_score` flag-inversion gotcha.

- [ ] **Step 2: Verify the README code blocks compile mentally against the signatures** (no test) and commit

```bash
cd /Users/jcastellanos/repos/proteinmpnn-ios
git add MPNNKit/README.md
git commit -m "docs(mpnnkit): document score()/design()/repack() primitives"
```

---

## Self-Review

**1. Spec coverage** (each §2 goal → task):
- `score()` + 3 modes + `currentAALogProb` → Tasks 6–7. ✓
- `design()` fixed/bias/omit + logits → Task 3. ✓
- standalone `repack()` + per-atom confidence → Task 4. ✓
- parity coverage → Task 2 (design path), Task 4 (repack), Tasks 6–7 (score). ✓
- MLX-free boundary → all public methods return `String`/`[[Float]]`/`[Int]`; `#if DEBUG` test helper is internal, not public. ✓
- `run()` bit-stable → Task 5 characterization test. ✓
- `MPNNInputError` + validation → Task 1 (type) + enforced in Tasks 3/4/7. ✓
- `alphabet` exposed → Task 1. ✓
- perf reality-check (spec §8 Tier 3) → **not yet a task**; add as a follow-up measurement task if desired (it is a measurement, not a code change; the L-sweep can be a `swift test` that prints `designMs`/`repackMs`/score timings at L for 6MRR + a large synthetic fixture). Noted as an optional Task 9.

**2. Placeholder scan:** The `...` in `capture_score.py` (Task 6, Step 1) are explicit "fill from `capture_design.py`'s model load / reference signatures shown in the Task 6 header" instructions with the surrounding real code and exact reference method names/line numbers — not vague TODOs. Swift steps contain complete code. The golden values in Task 5 are captured by an executable step, not left blank.

**3. Type consistency:** `designWeights`/`packerWeights` accessors (Task 2) are used by Tasks 3/7. `repackCore` (Task 4) is used by Task 5. `ScoreMode` (Task 1) is used by `MPNNInputError` (Task 1) and `score` (Task 7). `buildLogitBias`/`toRows`/`modelInputs` (Task 1) used throughout. Oracle key names (`logprobs_conditional`/`_unconditional`/`_leaveoneout`, `native_seq`, `decoding_order`) match between Task 6 (writer) and Task 7 (reader). Fixture input keys (`X`, `R_idx`, `chain_labels`, `S_native`) match the documented fixture schema.

---

## Optional Task 9 (measurement, no shipped code change)

Add `MPNNKit/Tests/MPNNKitTests/PerfTests.swift` that runs `score`/`design`/`repack` at L for 6MRR (68) and a large synthetic input, printing `designMs`/`repackMs` and score wall-time, and asserting only that they complete. This surfaces the spec §9 O(L²) reality (`fixedPositions` trims only decode work; the encode is whole-protein) without gating CI on device-specific timings. Run with `swift test --filter PerfTests`.
