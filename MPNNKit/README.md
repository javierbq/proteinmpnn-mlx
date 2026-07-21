# MPNNKit

On-device LigandMPNN **protein sequence design + side-chain repack** as a self-contained
Swift package (mlx-swift / Metal). Runs on backbone coordinates the host already has in
memory and returns a designed sequence + a repacked PDB. Extracted from the validated
MPNNBench inference core; numerically faithful to PyTorch/OpenFold (design top-1 100%,
repack side-chain RMSD ~1e-6 A).

## Install into an Xcode app (e.g. RayMol)

MPNNKit is a normal SPM package — add it the same way RayMol already adds its local
packages (*File > Add Package Dependencies... > Add Local...* -> this directory), or in a
`project.yml` / `Package.swift`:

```swift
.package(path: "../proteinmpnn-ios/MPNNKit")   // sibling-repo checkout
// target dependency: .product(name: "MPNNKit", package: "MPNNKit")
```

It pulls **mlx-swift** transitively (source-built, Metal kernels; SPM-cached after the first
resolve). All `xcodebuild` invocations for a host that includes it need
`-skipPackagePluginValidation -skipMacroValidation` (mlx-swift's `CudaBuild` plugin).

## The model is a separate, installable `.mpnnpack`

Code and weights are decoupled. The model ships as `MPNN.mpnnpack` (built by
`../port/build_mpnnpack.py`): a versioned, sha256-checked bundle of the two safetensors
weight files + geometry constants + atom-name table (~25 MB fp32). Ship it inside the app
bundle as a default, and/or let users drop a newer `.mpnnpack` into
`~/Library/Application Support/RayMol/models/` — new models then load without an app rebuild.

## Usage

```swift
import MPNNKit

let model = try MPNNModel(packDirectory: packURL)   // validates manifest + sha256, loads weights

let residues = structure.map {                       // from RayMol's in-memory atoms
    MPNNModel.Residue(n: $0.n, ca: $0.ca, c: $0.c, o: $0.o, chain: $0.chainIndex, resSeq: $0.resSeq)
}
var opts = MPNNModel.Options()
opts.temperature = 0.1      // 0 = greedy
opts.repack = true
let out = try model.run(residues, options: opts)

out.sequence                 // designed 1-letter sequence
out.pdb                      // repacked structure (PDB text) — load back as a new object
out.designMs / out.repackMs  // timings
```

No MLX types cross the public API (coords in via `SIMD3<Float>`, results out as `String`),
so the host doesn't need to `import MLX`. Call `run(_:)` off the main thread; a ~1k-residue
complex is a few seconds on an A17-class GPU.

## Composable primitives

`run()` is a convenience wrapper around three primitives you can call independently to build
custom pipelines (scoring designed variants, repacking with a fixed sequence, etc.). All three
are synchronous; call them off the main thread — a ~1 k-residue call is a few seconds on A17.

### Alphabet convention

```swift
MPNNModel.alphabet
// ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X"]
// 21 characters; index 20 = 'X' (unknown/mask).
// Column order of every [L,21] output (logProbs, logits, bias, omit, etc.).
```

### `score(_:sequence:mode:seed:)` -> `ScoreResult`

Per-position log-probabilities in one of three modes. `logProbs` is `[L][21]` (log-softmax);
`currentAALogProb` (length L) is the diagonal entry for the supplied sequence and is present
**iff** `sequence` is given. **All three modes are fully deterministic; the `seed` parameter
is accepted for API compatibility but does not affect score output.**

| Mode | Decode order | `sequence` required? | Semantics |
|------|-------------|----------------------|-----------|
| `.conditional` | Fixed identity `[0, 1, ..., L-1]` | Yes | Teacher-forced: position `i` conditioned on native residues at positions 0..`i`-1 |
| `.unconditional` | Strictly zeroed backward mask (structure only) | No | Marginals from backbone geometry alone; no sequence conditioning |
| `.leaveOneOut` | Canonical per-position: others in identity order, position `i` decoded last | Yes | `p(AA_i | structure + all other native residues)` - runs L separate decode passes; costlier |

```swift
// Conditional: log-prob of the native sequence under the fixed identity order
let native: [Int] = ...                        // alphabet indices, length L
let s = try model.score(residues, sequence: native, mode: .conditional)
s.logProbs           // [[Float]], shape [L][21]
s.currentAALogProb   // [Float]?, length L - log-prob of the actual AA at each position

// Unconditional: structure-only; no sequence needed
let u = try model.score(residues, mode: .unconditional)
u.currentAALogProb   // nil (no sequence supplied)

// Leave-one-out: how probable is each position given all other native residues?
let l = try model.score(residues, sequence: native, mode: .leaveOneOut)
l.currentAALogProb   // [Float]?, length L
```

### `design(_:options:)` -> `DesignResult`

Samples (or greedily decodes at `temperature: 0`) a new sequence for the backbone.
Returns `sequence` (1-letter String), `indices` ([Int], alphabet indices), and
`logits` ([[Float]], L x 21, pre-softmax at each decode step).

```swift
var opts = MPNNModel.DesignOptions()
opts.temperature = 0.1          // 0 -> greedy (deterministic)
opts.seed = 42                  // seeds decoding order + sampling; nil -> nondeterministic

// Fix positions 3 and 7 to native identity
opts.nativeSequence = native    // [Int], length L - required when fixedPositions non-empty
opts.fixedPositions = [3, 7]    // 0-based; held to nativeSequence, skipped in decode

// Palette restriction (both can be combined):
//   bias  - L x 21 additive logit matrix; -1e9 hard-excludes, negative values soft-penalize
//   omit  - length-L array of disallowed index sets (adds -1e9 internally per entry)
opts.bias = myBias              // [[Float]]?, shape [L][21]
opts.omit = myOmit              // [Set<Int>]?, length L

let d = try model.design(residues, options: opts)
d.sequence    // String, 1-letter AA in input residue order
d.indices     // [Int], alphabet indices
d.logits      // [[Float]], shape [L][21]
```

**Hard-assign a residue:** add its 0-based index to `fixedPositions` and supply its alphabet
index in `nativeSequence`. The decoder skips that position; the result holds the native AA there.

**Palette-restrict:** set `-1e9` bias entries in `bias`, or list disallowed indices in `omit`.

### `repack(_:sequence:)` -> `RepackResult`

Fixed-backbone side-chain packing for an externally-supplied sequence. Fully deterministic.

```swift
let r = try model.repack(residues, sequence: designedIndices)
r.pdb               // String - PDB text of the repacked structure
r.atomConfidence    // [[Float]], shape [L][14] - packer log-prob per atom14 slot
```

`atomConfidence[i][j]` is the packer log-probability for atom slot `j` of residue `i` in
atom14 layout. Slots beyond the residue's atom count are zero-padded.

### Regenerating parity oracles

Score fixtures are produced by `port/capture_score.py`, which runs the same LigandMPNN
`ProteinMPNN` weights against Python reference implementations of all three decode passes.

**LigandMPNN `single_aa_score` flag-inversion:** the `single_aa_score(use_sequence)` method
in `model_utils.py` has its `use_sequence` flag wired opposite to its help text
(lines 502-507): `True` strips the sequence (unconditional); `False` uses it (conditional).
`capture_score.py` avoids this by reimplementing the three decode passes directly over
`model.encode()`, mirroring the canonical ProteinMPNN `forward`/`unconditional_probs`/
`conditional_probs` from `protein_mpnn_utils.py` instead.

---

## Notes

- **Protein-only**: design is canonical ProteinMPNN (ligand context zeroed); all positions are designed.
- Featurization/kNN run in **fp32** (fp16 reorders the neighbor graph); this is internal and automatic.
- Sources `MLXCore/MPNNLayers/DesignModel/PackerModel/Geometry/RepackLoop/PDBWriter` are the
  same code validated in MPNNBench; `MPNNPack`/`MPNNModel` are the package's public surface.
