# MPNNKit

On-device LigandMPNN **protein sequence design + side-chain repack** as a self-contained
Swift package (mlx-swift / Metal). Runs on backbone coordinates the host already has in
memory and returns a designed sequence + a repacked PDB. Extracted from the validated
MPNNBench inference core; numerically faithful to PyTorch/OpenFold (design top-1 100%,
repack side-chain RMSD ~1e-6 Å).

## Install into an Xcode app (e.g. RayMol)

MPNNKit is a normal SPM package — add it the same way RayMol already adds its local
packages (*File ▸ Add Package Dependencies… ▸ Add Local…* → this directory), or in a
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

## Notes

- **Protein-only**: design is canonical ProteinMPNN (ligand context zeroed); all positions are designed.
- Featurization/kNN run in **fp32** (fp16 reorders the neighbor graph); this is internal and automatic.
- Sources `MLXCore/MPNNLayers/DesignModel/PackerModel/Geometry/RepackLoop/PDBWriter` are the
  same code validated in MPNNBench; `MPNNPack`/`MPNNModel` are the package's public surface.
