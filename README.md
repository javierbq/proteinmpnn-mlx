# ProteinMPNN → iOS: export/validation harness

Companion to `../ProteinMPNN-iOS-feasibility.md`. Proves ProteinMPNN's static
scoring path exports to Core ML with numerical parity, before any Swift is written.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy coremltools
```

## Milestone 0 — reference oracle (PyTorch ground truth)
```bash
.venv/bin/python milestone0_oracle.py \
  --pdb ../ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb \
  --weights ../ProteinMPNN/vanilla_model_weights/v_48_020.pt
# -> oracle_6MRR.npz  (inputs + reference log-probs + a sampled design)
```

## Milestone 1 — Core ML export + parity check
```bash
.venv/bin/python milestone1_coreml_export.py --oracle oracle_6MRR.npz --precision fp32
# -> ProteinMPNN_uncond.mlpackage ; prints CoreML-vs-oracle max|diff|
```

## Verified results (2026-07-18, torch 2.13 / coremltools 9.0, macOS arm64)
- Model = 1,660,485 params; oracle `sample()` runs on CPU.
- Export recipe: `torch.export.export` → `.run_decompositions({})` → `ct.convert(convert_to="mlprogram")`.
  Pin `topk` k to a constant (stock `np.minimum(top_k, X.shape[1])` is export-hostile).
- **fp32: max|diff| = 2.5e-5, 100% top-1 AA agreement (PASS).**
- **fp16: BROKEN (max|diff| 3.6, 34% agreement)** — fp16 CA–CA distances change the
  `topk` kNN graph. Keep featurization fp32, or precompute `E_idx` on the host.

## Repacking reference (LigandMPNN side-chain packing)
```bash
.venv/bin/python repack_oracle.py --pdb ../LigandMPNN/inputs/1BC8.pdb
# -> repack_ref.npz : per-step von-Mises mean/concentration/mix_logits + atom14 coords
```
Verified: Packer = 3,571,556 params; 3 denoising steps captured; final atom14 `[1,93,14,3]`.

## iOS Simulator test (Swift + Core ML)
```bash
.venv/bin/python export_sim_fixtures.py                 # writes sim_fixtures/*.bin
xcrun coremlcompiler compile ProteinMPNN_uncond.mlpackage .
SDK=$(xcrun --sdk iphonesimulator --show-sdk-path)
xcrun --sdk iphonesimulator swiftc -target arm64-apple-ios18.0-simulator -sdk "$SDK" \
  sim_test/main.swift -o sim_test/sim_test -framework CoreML -framework Foundation
UDID=$(xcrun simctl create pmpnn "iPhone 16 Pro" com.apple.CoreSimulator.SimRuntime.iOS-26-4)
xcrun simctl boot "$UDID"; xcrun simctl bootstatus "$UDID" -b
xcrun simctl spawn "$UDID" "$PWD/sim_test/sim_test" "$PWD/ProteinMPNN_uncond.mlmodelc" "$PWD/sim_fixtures"
```
Verified in iOS 26.4 Simulator: max|diff| 1.9e-5, 100% top-1 agreement.
NOTE: the Simulator has no Metal/MPS backend (falls back to CPU), so its latency
(~232 ms) is not representative of on-device GPU/ANE speed — profile on real hardware.

## Cross-protein stability benchmark
```bash
.venv/bin/python benchmark_proteins.py     # -> benchmark_results.json
```
6 proteins, 68-960 residues, 1-6 chains. Result: **fp32 = 100% top-1 & max|diff| < 4e-5
for every protein** (no failures); **fp16 breaks on all (14-68% top-1), worse as L grows.**
fp32 latency (macOS Core ML runtime, not device): 5 ms @68 res → 57 ms @960 res.

## End-to-end design -> repack benchmark (OpenFold geometry)
```bash
.venv/bin/python benchmark_design_repack.py    # -> benchmark_design_repack.json
```
LigandMPNN design then side-chain repack (OpenFold torsion/frame denoiser, 3 steps x 16 samples),
6 proteins 68-955 res. Result: **100% side-chain completeness, NaN-free, no failures on all.**
PyTorch/CPU reference timing: design 119->1345 ms, repack 343->2452 ms (68->955 res).
NOTE: repack is the PyTorch reference; it is NOT yet Core ML-exported / natively ported.

## Flexible-length export ([milestone2_flexible_export.py](milestone2_flexible_export.py))
```bash
.venv/bin/python milestone2_flexible_export.py
```
- Single dynamic-length model (`RangeDim`/`EnumeratedShapes`) **FAILS** to convert:
  coremltools 9 gather rank bug under symbolic shapes ("Rank mismatch: input rank 4, indices rank 3").
- **Per-length bucket models WORK.** Verified 6MRR(L=68) through the L=106 bucket model with
  padding: max|diff| 1.9e-5, 100% top-1. Deploy = a few fixed bucket models + pad-to-nearest.

## Repack + OpenFold geometry export probe ([repack_export_probe.py](repack_export_probe.py))
```bash
.venv/bin/python repack_export_probe.py
```
- OpenFold geometry (torsions -> Rigid frames -> atom14): `torch.export` OK, but
  **coremltools convert FAILS** — "Core ML only supports tensors with rank <= 5"
  (rigid-group frame math is rank 6). Repack geometry must be **native** (Swift/Metal)
  or **MLX** (no rank cap); it cannot be a pure Core ML export.

## Not yet done
- Full autoregressive sampling on-device (host-driven loop — see feasibility report §3/§5).
- **On-device** (physical iPhone) latency profiling — the Simulator has no Metal (CPU fallback).
- Native (MLX/Metal) reimplementation of the repacking OpenFold geometry vs `repack_ref.npz`.
