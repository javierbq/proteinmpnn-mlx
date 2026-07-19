# MPNNBench iOS Implementation Plan

> **For agentic workers:** Execute phase-by-phase. Each task ends with a concrete validation
> gate (a command + expected result). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build and install an iOS app that runs LigandMPNN sequence design + side-chain
repack on-device, writes a PDB, and benchmarks latency + peak memory across protein sizes.

**Architecture:** Offline Python first establishes an MLX design port + weight/fixture/oracle
exports (the Swift spec + parity oracle). Then transcribe the validated MLX-Python (features,
layers, design, packer, geometry) to mlx-swift, wrap in a SwiftUI benchmark harness, sign, and
install to the physical iPhone 15 Pro.

**Tech Stack:** Python 3.12 (torch 2.13, mlx 0.32, numpy 2.5) · Swift 6 / mlx-swift 0.31.6 ·
SwiftUI · Xcode 26.4 · `xcrun devicectl`.

## Global Constraints

- Featurization + kNN graph MUST run in **fp32** (fp16 reorders the `topk` graph → 34% top-1).
- Design model: `../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt` (2.62M params).
- Packer model: `../LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt` (3.57M params).
- Protein-only: ligand/nucleic context zeroed everywhere.
- Parity uses deterministic paths only (fixed decoding order + greedy design; mode repack).
- Bundle id `com.javiercv.mpnnbench`; signing team `R7QJV4RY95`; min iOS 18.
- MLX-Python files under `port/` are the canonical Swift spec — transcribe, don't reinvent.
- Real PDBs: `../ProteinMPNN/inputs/{PDB_monomers,PDB_homooligomers,PDB_complexes}/pdbs/*.pdb`.
- Weights bundled as safetensors via `mx.save_safetensors` (fp32).

---

## Phase 0 — Offline Python foundation (critical path)

### Task 0.1: MLX-Python design port

**Files:**
- Create: `port/mlx_design.py` (LigandMPNN design: featurize → encode → AR decode → W_out)
- Create: `port/capture_design.py` (PyTorch oracle capture: greedy + fixed decoding order)
- Create: `port/validate_mlx_design.py` (MLX vs PyTorch parity)

**Interfaces:**
- Consumes: `port/mlx_features.py` (featurization, already validated), `port/mlx_packer.py`
  (Enc/Dec layer patterns + weight-loader idioms).
- Produces: `mlx_design.design_greedy(X, mask, residue_idx, chain_enc, decoding_order, weights)
  -> (S[L] int32, logits[L,21])` and `design_sample(..., temperature, key)`.

- [ ] Read `../LigandMPNN/model_utils.py` (class `ProteinMPNN`, `sample`, `forward`) + how
  `run.py` builds features for the design model; confirm the encoder/decoder layer math equals
  the packer's (it does — same `EncLayer`/`DecLayer` op family) so MLX layers are reused.
- [ ] Implement `mlx_design.py`: reuse `mlx_features` featurization; encoder ×3; the
  single-step decoder + `W_out` head; a host `decoding_order` loop with greedy `argmax` and a
  `mx.random.categorical` sampling variant. Ligand context zeroed.
- [ ] PyTorch→MLX weight loader for `ligandmpnn_v_32_010_25.pt` (mirror `mlx_packer`'s loader).
- [ ] `capture_design.py`: load the PyTorch LigandMPNN design model, run on 6MRR with a
  **fixed decoding order** and greedy selection; dump `S`, `logits[L,21]`, decoding order,
  and the featurizer tensors → `port/design_capture.npz`.
- [ ] `validate_mlx_design.py`: compare MLX greedy vs the capture.
- **GATE:** `.venv/bin/python port/validate_mlx_design.py` → greedy top-1 == 100%,
  `max|logits diff|` < 1e-4 on 6MRR (and 5L33). Fix until it passes.

### Task 0.2: Weight conversion → safetensors

**Files:**
- Create: `port/convert_weights.py`

- [ ] Convert both `.pt` files to MLX-layout safetensors:
  `app_assets/design.safetensors`, `app_assets/packer.safetensors` (fp32). Reuse the exact
  key-rename/transpose maps from `mlx_design`/`mlx_packer` loaders so Swift loads them directly.
- [ ] Round-trip check: load safetensors back into the MLX modules, re-run the 0.1 parity gate.
- **GATE:** re-run `validate_mlx_design.py` and `port/validate_mlx_repack.py` loading from the
  safetensors files → same parity numbers as the `.pt` path.

### Task 0.3: Fixture + oracle export (all ladder entries)

**Files:**
- Create: `port/export_app_assets.py`
- Create: `app_assets/` (bundled resources)

- [ ] For each real protein (6MRR, 5L33, 4GYT, 3HTN, 4YOW, 6EHB): parse the PDB (biopython/
  prody, offline), extract `X[L,4,3]`, `residue_idx` (with +100 chain gaps), `chain_encoding`,
  chain letters, native sequence → write `app_assets/inputs/<id>.pmb` (format in spec §5).
- [ ] Synthetic ceiling entries: tile 5L33 (106 res) with +100 gaps to ~1200/1500/2000/2500
  residues → `app_assets/inputs/synthN.pmb`, flagged `synthetic`.
- [ ] For each real protein: run the deterministic **design (greedy, fixed order)** and
  **repack (mode)** through the validated MLX-Python pipeline; dump oracle `top1[L]`,
  `logits[L,21]`, `atom14[L,14,3]`, and the decoding order → `app_assets/oracles/<id>.oracle`.
- [ ] Dump geometry constant tables (`port/geometry_constants.npz`) → `app_assets/geometry.bin`.
- [ ] Write `app_assets/manifest.json`: list of entries with `id, L, n_chains, synthetic, hasOracle`.
- **GATE:** a re-loader script reads every `.pmb`/`.oracle` back and prints shapes; counts match
  the manifest; `python` re-parse of one exported input reproduces L and sequence.

---

## Phase 1 — Swift MLX core (transcription + fixture parity)

### Task 1.1: Xcode project + mlx-swift + smoke test

**Files:**
- Create: `MPNNBench/` (Xcode iOS app project, SwiftUI), `MPNNBench.xcodeproj`
- Create: `MPNNBench/MPNNBenchApp.swift`, `ContentView.swift` (placeholder)

- [ ] Generate the iOS app project (via `xcodegen`/`xcodebuild -create` or a hand-written
  `project.yml` + xcodegen; fall back to a template project). Add SPM dependency
  `github.com/ml-explore/mlx-swift` @ 0.31.6 (products: `MLX`, `MLXNN`, `MLXRandom`).
  Bundle id `com.javiercv.mpnnbench`, team `R7QJV4RY95`, min iOS 18, Metal enabled.
- [ ] Add `app_assets/` as a bundled resource folder reference.
- **GATE:** `xcodebuild -project MPNNBench.xcodeproj -scheme MPNNBench
  -destination 'generic/platform=iOS' build` succeeds; a trivial MLX op
  (`MLXArray([1,2,3]).sum()`) compiles and links against mlx-swift.

### Task 1.2: InputStore + Weights loader (Swift)

**Files:**
- Create: `MPNNBench/Core/InputStore.swift`, `MPNNBench/Core/Weights.swift`
- Create: `MPNNBenchTests/InputStoreTests.swift`

- [ ] `InputStore.load(id) -> ProteinInput` parsing the `.pmb` format; `oracle(id) -> OracleRef`.
- [ ] `Weights.load(name) -> [String: MLXArray]` via `MLX.loadArrays(url:)` (safetensors).
- **GATE (macOS unit test):** load 6MRR `.pmb` → `L == 68`, sequence length 68; load
  `design.safetensors` → expected key count + a spot-checked tensor shape.

### Task 1.3–1.7: Transcribe MLX modules with fixture parity

For each, transcribe the named Python file to Swift mlx-swift (same array API), then a unit
test feeds the **bundled fixture inputs** and asserts the Swift output matches the Python MLX
output (exported alongside fixtures) within tolerance.

- [ ] **1.3 Featurizer** ← `port/mlx_features.py` → `MPNNBench/Core/Featurizer.swift`.
  GATE: `E_idx` exact match; all feature tensors `max|diff| < 1e-4` vs Python on 6MRR/5L33.
- [ ] **1.4 MPNNLayers** ← Enc/Dec layers in `port/mlx_packer.py` → `MPNNBench/Core/MPNNLayers.swift`.
  GATE: `h_V`/`h_E` after encode `max|diff| < 1e-4`.
- [ ] **1.5 DesignModel** ← `port/mlx_design.py` → `MPNNBench/Core/DesignModel.swift`.
  GATE: greedy `top1` == oracle 100%, `max|logits diff| < 1e-3` on 6MRR/5L33.
- [ ] **1.6 Geometry** ← `port/mlx_geometry.py` (+ `geometry.bin`) → `MPNNBench/Core/Geometry.swift`.
  GATE: atom14 from fixed torsions `max|diff| < 1e-3 Å` vs Python.
- [ ] **1.7 PackerModel** ← `port/mlx_packer.py` → `MPNNBench/Core/PackerModel.swift`.
  GATE: mode-repack atom14 RMSD `< 1e-2 Å` vs oracle on 6MRR/5L33.

---

## Phase 2 — App harness (PDB, parity, benchmark, UI)

### Task 2.1: PDBWriter

**Files:** Create `MPNNBench/Core/PDBWriter.swift`, `MPNNBenchTests/PDBWriterTests.swift`
- [ ] `PDBWriter.write(backbone, atom14, atom14Mask, sequence, chainLetters, residueIdx) -> String`
  emitting valid `ATOM` records (serial, atom name, resName, chain, resSeq, x/y/z, occ=1.00,
  bfactor, element).
- **GATE:** generated PDB for 6MRR re-parses offline (prody) → residue count 68, 100%
  expected side-chain atoms present, no NaN coords.

### Task 2.2: ParityChecker

**Files:** Create `MPNNBench/Core/ParityChecker.swift`, tests.
- [ ] `check(id, S, logits, atom14) -> ParityResult { designTop1Pct, maxLogitDiff, repackRmsd, pass }`
  comparing against `InputStore.oracle(id)`.
- **GATE (unit):** feeding the oracle's own values yields `pass == true`, 100%, ~0 diffs.

### Task 2.3: Benchmark harness (timing + memory sampler)

**Files:** Create `MPNNBench/Core/Benchmark.swift`, `MPNNBench/Core/MemorySampler.swift`.
- [ ] `MemorySampler`: background thread polling `task_vm_info.phys_footprint` every ~20 ms →
  peak/baseline/delta MB; `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` handler.
- [ ] `Benchmark.run(inputs, mode) -> [BenchResult]`: per protein warm-up once, then timed
  featurize/encode/design/repack/pdb-write with the memory sampler active; assemble `BenchResult`.
- **GATE (macOS):** `Benchmark.run([6MRR])` returns finite latencies, `peak_footprint_mb > 0`,
  and (deterministic mode) `parity_pass == true`.

### Task 2.4: SwiftUI BenchView + export

**Files:** Create `MPNNBench/BenchView.swift`; replace `ContentView.swift`.
- [ ] List of bundled proteins (id, L, chains, synthetic badge); "Run sweep" button; per-row
  live progress; results table (latencies + peak MB + parity ✓/✗); share-sheet export of the
  aggregated JSON and the generated PDB(s).
- **GATE (macOS run / Simulator):** app launches, "Run sweep" over the small proteins fills the
  table with parity ✓ and nonzero timings; export produces a JSON file + a PDB file.

---

## Phase 3 — Device build + install

### Task 3.1: Device build & signing

- [ ] Resolve automatic signing for team `R7QJV4RY95`; register the device if needed
  (automatic signing handles UDID `00008130-00040D8C01F0001C`).
- **GATE:** `xcodebuild -destination 'platform=iOS,id=<devicectl-id>' build` produces a signed
  `MPNNBench.app`.

### Task 3.2: Install to iPhone 15 Pro

- [ ] Confirm the phone is connected + unlocked (currently offline — user brings it online).
- [ ] `xcrun devicectl device install app --device <id> <path>/MPNNBench.app`.
- [ ] Guide first-launch dev-cert trust (Settings ▸ General ▸ VPN & Device Management).
- **GATE:** app installs and launches on device; run the parity gate on-device (expect all ✓),
  then the full sweep; capture the results JSON.

### Task 3.3: Results capture

- [ ] Pull the exported JSON + a sample generated PDB off the device (share sheet → Files/AirDrop,
  or `devicectl device copy from`); update `README.md` with the on-device latency + memory table
  (the first real-hardware numbers — the feasibility report's open item §11.1).

---

## Self-review notes

- **Spec coverage:** §3 pipeline → Tasks 1.3–1.7 + 2.1; §6 metrics → 2.3; §7 parity → 0.3/2.2;
  §8 ladder → 0.3; §9 offline prereqs → Phase 0; §10 transcription → Phase 1; §11 install → Phase 3.
- **Determinism:** every parity gate uses fixed-order greedy / mode (matches spec §7).
- **fp32 constraint** carried into Task 1.3 gate (E_idx exact).
- **No new heavy deps** on device beyond mlx-swift (spec §2/§14).
