# MPNNBench — on-device ProteinMPNN design + side-chain repack benchmark (iOS)

_Design doc / spec. Date: 2026-07-18. Companion to `../../../README.md`, `port/README.md`,
and `../../../../ProteinMPNN-iOS-feasibility.md`._

## 1. Goal

A single-purpose iOS app that runs the real **sequence design → side-chain repack →
write-PDB** pipeline entirely on-device (Apple Silicon, MLX/Metal), across a ladder of
input proteins of increasing size, and reports **latency and peak memory per size**. A
built-in **parity gate** proves the on-device port is numerically correct against captured
oracle references before any timing number is trusted. Deliverable: the app built and
installed on the user's iPhone 15 Pro.

This is the culmination of the prior feasibility + porting work: ProteinMPNN-family design
is validated (Core ML, exact parity) and the LigandMPNN side-chain packer (NN + OpenFold
geometry) is already ported to MLX-Python and validated to ~1e-5 vs PyTorch. What remains is
(a) an MLX design port, (b) transcription of the validated MLX-Python to mlx-swift, and
(c) a benchmark app + on-device install.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Runtime | **MLX-Swift, unified** | One runtime for design + repack; native autoregressive loop; dynamic length (no Core ML size buckets); repacker geometry cannot be pure Core ML (rank-6); reuses validated MLX-Python port directly. |
| Design model | **LigandMPNN design `ligandmpnn_v_32_010_25.pt`** (2.62M params) | Matched pair with the sc packer; shares featurization with `port/mlx_features.py`. |
| Repack model | **LigandMPNN sc packer `ligandmpnn_sc_v_32_002_16.pt`** (3.57M params) | Already ported + validated in `port/`. |
| App scope | **Benchmark harness** (bundled inputs, no import UI, no on-device PDB parser) | Matches the stated goal; minimal surface. |
| Size range | **Validated 6 (68→955) + synthetic ceiling ladder (~1200/1500/2000+)** | Real apples-to-apples numbers + locate the O(L²) memory/latency limit. |
| Bundle id / signing | `com.javiercv.mpnnbench`, team **R7QJV4RY95** (`javiercv@uw.edu`, Apple Development) | User's personal identifier; only iOS-capable identity present. |
| App name | **MPNNBench** | — |
| Min iOS | 18.0 (device runs 26.5.2) | mlx-swift Metal support. |

## 3. On-device pipeline (per protein)

1. **Load input.** Bundled compact binary per protein: backbone coords `X = N/CA/C/O
   [L,4,3]` (fp32), `residue_idx [L]` (int, with the +100 inter-chain gap already applied),
   `chain_letter [L]`, `chain_encoding [L]`, and the native sequence string. Pre-extracted
   offline → no on-device PDB parser.
2. **Featurize** (MLX, **fp32**). Backbone → virtual Cβ → CA–CA distance matrix → `topk`
   kNN graph `E_idx [L,K]` → 25-atom-pair RBF features + relative-position one-hot →
   `Linear/LayerNorm`. Ligand context zeroed (protein-only). Transcribed from
   `port/mlx_features.py`. **fp32 is mandatory** — fp16 changes CA–CA distances enough to
   reorder the `topk` graph (documented: 34% top-1 in fp16).
3. **Design** (LigandMPNN, MLX). Encoder (3 message-passing layers) runs once → `h_V, h_E`.
   Then a **native Swift autoregressive loop**: decoding order = `argsort(chain_mask·|randn|)`
   (or fixed for parity); per step gather neighbor state at position `t`, run 3 decoder
   layers on that one position → `logits[21]`, apply temperature/bias, sample
   (`mx.random.categorical`) or greedy `argmax`, write the chosen AA embedding into running
   state `h_S`, `S`. O(L) sequential steps. → designed sequence `S [L]`.
4. **Repack** (LigandMPNN sc packer, MLX). Re-featurize with the designed sequence (aatype);
   encode once; **denoising loop** (`num_denoising_steps = 3`): decode → von-Mises mixture
   (`mean/concentration/mix_logits` over χ1–χ4, 3 components) → **mode** (deterministic) or
   sample → torsions → global rigid frames → atom14 coords via `port/mlx_geometry.py` +
   constant tables → feed updated coords into the next step. → side-chain atom14 `[L,14,3]`.
5. **Write PDB.** Backbone (N/CA/C/O) + packed atom14 side-chain atoms + designed sequence →
   valid `ATOM` records with chain IDs, residue numbers, element column, occupancy 1.00,
   b-factor from packer confidence. Saved to app Documents; exported via the system share
   sheet.

## 4. Module boundaries

Each is independently testable; MLX modules mirror the Python files 1:1 (Python = Swift spec).

| Module | Responsibility | Key interface |
|---|---|---|
| `InputStore` | Load bundled protein inputs + oracle refs | `func load(id) -> ProteinInput`, `func oracle(id) -> OracleRef` |
| `Featurizer` | coords → kNN graph + node/edge features (fp32) | `func encode(X, mask, residueIdx, chainEnc) -> Features` |
| `MPNNLayers` | `EncLayer`, `DecLayer` building blocks (shared by design + packer) | MLX modules |
| `DesignModel` | LigandMPNN design encoder + AR decoder + `W_out`; `sample`/`greedy` | `func design(Features, opts) -> (S, logProbs)` |
| `PackerModel` | LigandMPNN sc encoder + decoder + `W_torsions`; von-Mises mode/sample | `func repack(X, S, opts) -> Atom14` |
| `Geometry` | torsion→frame→atom14 + constant tables | `func atom14(frames, torsions, aatype) -> coords` |
| `PDBWriter` | (backbone, atom14, sequence, meta) → PDB text | `func write(...) -> String` |
| `Benchmark` | stage timing + background memory sampler; drives the sweep | `func run([ProteinInput]) -> [BenchResult]` |
| `ParityChecker` | compare deterministic on-device outputs vs bundled oracle | `func check(id, S, atom14) -> ParityResult` |
| `Weights` | load safetensors → MLX arrays into modules | `func load(name) -> [String: MLXArray]` |
| `BenchView` | SwiftUI: list, Run sweep, progress, results table, parity, export | — |

## 5. Data formats

- **Input binary** (`.pmb`, one per protein): little-endian header (`L`, `nChains`) then
  `X` fp32 `[L*4*3]`, `residue_idx` int32 `[L]`, `chain_encoding` int32 `[L]`, `chain_letter`
  ascii `[L]`, native sequence ascii `[L]`. Written by an offline Python exporter.
- **Oracle ref** (`.npz`-equivalent, converted to a bundled binary): deterministic design
  `top1 [L]` + `logProbs [L,21]` (fixed decoding order, greedy), and repack `atom14 [L,14,3]`
  (mode-selected, RNG-free), plus the fixed decoding order used.
- **Weights** (`.safetensors`): `design.safetensors` (from `ligandmpnn_v_32_010_25.pt`) and
  `packer.safetensors` (from `ligandmpnn_sc_v_32_002_16.pt`), fp32, keys/transposes matched
  to the MLX module layout. ~10.5 + 14.3 MB.
- **Geometry constants**: `port/geometry_constants.npz` → bundled binary tables
  (`restype_rigid_group_default_frame`, `restype_atom14_rigid_group_positions`,
  `restype_atom14_to_rigid_group`, `restype_atom14_mask`).

## 6. Benchmark metrics

Recorded per protein into `BenchResult`:
- **Latency** (mach_absolute_time / `ContinuousClock`): `featurize_ms`, `encode_ms`,
  `design_ms` (+ `design_ms_per_residue`), `repack_ms` (+ `repack_ms_per_step`),
  `pdb_write_ms`, `total_ms`. Warm run after one warm-up to exclude first-call compile.
- **Memory**: `task_vm_info.phys_footprint` sampled every ~20 ms on a background thread →
  `peak_footprint_mb`, `baseline_mb`, `delta_mb`. Register a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`
  handler; record if the OS warns/terminates near the top of the ladder.
- **Correctness**: `design_top1_agreement` (%), `repack_atom14_rmsd` (Å), `parity_pass` (bool).
- **Meta**: `L`, `n_chains`, `synthetic` (bool), device model, iOS version, MLX version.
- Export: on-screen table + one-tap JSON (all `BenchResult`) + the generated PDB(s).

## 7. Parity methodology

RNG will not match cross-platform, so parity uses **deterministic** paths:
- Design: **fixed decoding order** (bundled in the oracle) + **greedy `argmax`**. Pass if
  on-device `top1` matches oracle `top1` ≥ 99% and `max|logProb diff|` < 1e-3.
- Repack: **mode-selected** von-Mises (no sampling). Pass if atom14 RMSD < 1e-2 Å vs oracle.
- Timed sweep runs afterward and may use stochastic sampling (temperature) — realistic usage;
  latency is essentially identical to the deterministic path.

## 8. Size ladder

- **Real (validated):** 6MRR (68,1ch), 5L33 (106,1), 4GYT (354,2), 3HTN (425,3),
  4YOW (~681,3), 6EHB (~955,3).
- **Synthetic ceiling:** tiled backbones (a validated monomer replicated with +100
  residue-idx chain breaks) at ~1200 / 1500 / 2000 / (2500 if memory allows) residues.
  Clearly labeled `synthetic`. Purpose: locate where featurization O(L²) memory or per-step
  latency becomes the limiting factor on device (iPhone 15 Pro, 8 GB, jetsam-limited).

## 9. Offline Python prerequisites (in `.venv`, before Swift)

1. **MLX-Python design port** (`port/mlx_design.py`) — reuse `mlx_features` + the MLX
   Enc/Dec layers; add the autoregressive decode + `W_out` (21-way) head + host decoding-order
   loop; PyTorch→MLX weight loader for `ligandmpnn_v_32_010_25.pt`. Validate: greedy top-1 ==
   PyTorch oracle, `max|logProb diff|` ~1e-5 on the real 6 (same rigor as the repack port).
2. **Weight conversion** (`port/convert_weights.py`) — `.pt` → `.safetensors` in MLX layout
   for both models.
3. **Fixture + oracle generation** (`port/export_fixtures.py`) — input `.pmb` binaries + oracle
   refs (deterministic design + mode repack) + geometry constants for every ladder entry,
   including the synthetic tiled inputs.

## 10. Swift / mlx-swift transcription

New Swift package/modules transcribing the validated MLX-Python (`mlx_features`, MPNN layers,
`mlx_design`, `mlx_packer`, `mlx_geometry`) to mlx-swift (same array API). Correctness net:
**fixture-level unit tests** feeding identical inputs to Swift MLX and comparing intermediate
tensors against the Python MLX outputs (`h_V`, `h_E`, `E_idx`, logits, torsion params, atom14).

## 11. Xcode project, signing, install

- iOS app target (SwiftUI), **mlx-swift** SPM dependency (`github.com/ml-explore/mlx-swift`),
  Metal enabled, min iOS 18. Bundle weights + inputs + oracle refs + geometry constants as
  resources.
- Automatic signing, team `R7QJV4RY95`, bundle id `com.javiercv.mpnnbench`.
- Install to **Javier's iPhone 15 Pro** (`00008130-00040D8C01F0001C`) via
  `xcrun devicectl device install app`. Requires: phone connected + unlocked (currently
  offline), one-time "trust this computer," and trusting the developer cert under
  Settings ▸ General ▸ VPN & Device Management on first launch.

## 12. Testing strategy

1. **Python parity:** MLX design port vs PyTorch (greedy top-1, log-probs ~1e-5); repack
   already validated in `port/`.
2. **Swift↔Python MLX parity:** unit tests on bundled fixtures (intermediate-tensor compare).
3. **On-device parity gate:** in-app deterministic run vs bundled oracles → PASS/FAIL per
   protein, shown before timing numbers.
4. **On-device sweep:** warm-run latency + peak-memory ladder; export JSON + PDB.
5. **PDB sanity:** generated PDB re-parses (offline) and side-chain completeness == 100%.

## 13. Risks

- **mlx-swift op/latency parity** — the per-residue AR loop is L sequential MLX evals;
  host↔GPU sync per step may dominate latency. Minimize `eval()` sync points; this is itself
  a headline finding, to be measured, not assumed.
- **Swift↔Python numerical drift** — mitigated by fixture unit tests (§12.2).
- **Synthetic-large realism** — tiled inputs are not native folds; labeled `synthetic`; used
  only for the memory/latency curve, never for parity claims.
- **Memory ceiling / jetsam** — O(L²) featurization at 2k+ res; memory-pressure handler +
  graceful abort so the app records the limit instead of crashing.
- **Device gating** — phone must be online/unlocked at install; dev-cert trust on first launch.

## 14. Out of scope (YAGNI)

Import-your-own-PDB UI; on-device PDB parser; ligand/nucleic-acid context; tied/symmetric
design; PSSM/bias UI; Core ML path; ANE tuning; multi-model comparison UI. All deferrable and
not needed for the benchmark deliverable.

## 15. Sequencing (high level; detailed plan via writing-plans)

1. Offline Python: design MLX port + validation → weights → fixtures/oracles.
2. Swift MLX transcription + fixture unit tests (features → layers → design → packer → geometry).
3. PDBWriter + ParityChecker + Benchmark harness.
4. SwiftUI BenchView + JSON/PDB export.
5. Xcode project, signing, build, install to device; run parity gate + sweep; capture results.
