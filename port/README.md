# Repacking port (native-target)

Porting LigandMPNN side-chain repacking off OpenFold/PyTorch toward the iOS
native path (Swift/Metal or MLX). Built + validated incrementally against the
PyTorch+OpenFold reference.

## Why a port is needed
The OpenFold geometry (`torsion_angles_to_frames` -> `frames_and_literature_positions_to_atom14_pos`)
**cannot** be Core ML-exported: its `Rigid` frame math broadcasts to a **rank-6**
tensor and Core ML caps rank at 5 (see `../repack_export_probe.py`). So the
geometry must be reimplemented; the NN part is standard message passing.

## Status

| Piece | File | Status |
|---|---|---|
| Backbone frame from N/CA/C (Gram-Schmidt) | `geometry_port.py:make_backbone_frame_port` | ✅ matches OpenFold 1e-7 |
| torsion angles -> global rigid frames | `geometry_port.py:torsion_angles_to_frames_port` | ✅ |
| frames + ideal geom -> atom14 coords | `geometry_port.py:frames_to_atom14_port` | ✅ matches ~2e-6 Å |
| Full chain coords -> atom14 | validated end-to-end | ✅ ~4e-6 Å, 3 proteins |
| Constant tables | `geometry_constants.npz` | ✅ dumped for native side |
| **Geometry in MLX** (frame + torsions->atom14) | `mlx_geometry.py` | ✅ matches OpenFold ~4e-6 A (3 proteins) |
| **NN packer encode (msg-passing + ligand context)** | `mlx_packer.py:encode` | ✅ MLX, h_V 1.5e-6 / h_E 1e-5 |
| **NN packer decode + torsion head** | `mlx_packer.py:decode` | ✅ MLX, mean 4e-6 / mix 8e-6 / conc 2e-4 rel |
| **Featurization (features_encode/decode)** | `mlx_features.py` | ✅ MLX, E_idx exact + all outputs ~1e-6 |
| **Full forward coords->torsions** | `validate_mlx_features.py` | ✅ MLX end-to-end matches PyTorch |
| **Full pipeline coords->atom14 (deterministic)** | `validate_mlx_repack.py` | ✅ composes, 1.05e-5 A vs torch |
| Backbone torsion extraction (atom37_to_torsion_angles) | — | ⬜ TODO (3 standard dihedrals; taken from struct for now) |
| von Mises mixture mode/sampling | mode ✅ (validated); sampling | ⬜ TODO stochastic (host RNG) |
| multi-step denoising loop | — | ⬜ TODO (host: repeat decode->geometry->update X) |
| transcribe MLX Python -> mlx-swift | — | ⬜ TODO (mechanical, same API) |

**The whole packer NN + geometry is ported to MLX and validated to ~1e-5 vs PyTorch/OpenFold.**
What's left is mechanical: 3 backbone dihedrals, the host denoising loop, stochastic sampling,
and transcription to mlx-swift. Validated in Python (mlx 0.32) against captured fixtures
(`packer_capture.npz`); mlx-swift shares the API, so these files are the Swift spec.

## Reproducibility across multiple examples (fixed seed 111)
`validate_repack_multi.py` compares MLX vs the original PyTorch/OpenFold packer on
4 proteins (68-425 res, monomers/oligomers/complexes), deterministic mode:

| protein | L | shared-graph atom14 | own-graph E_idx diff | own-graph atom14 |
|---|---|---|---|---|
| 6MRR | 68 | 1.05e-5 A | 0 | 1.05e-5 A |
| 5L33 | 106 | 1.14e-5 A | 0 | 1.14e-5 A |
| 4GYT | 354 | 8.6e-6 A | 0 | 8.6e-6 A |
| 3HTN | 425 | 9.5e-6 A | 94/13600 (ties) | 1.5e-4 A |

- With a shared kNN graph, the ported compute is numerically exact (~1e-5) on EVERY example.
- End-to-end, outputs are nearly identical. 3HTN (a symmetric complex) has 94/13600 E_idx
  entries differ, ALL at exactly-equal neighbor distances (pure kNN tie-breaks, gap=0.00 A) ->
  sub-milli-Angstrom (1.5e-4 A) coordinate changes, physically negligible.

Run all validations:
```bash
../.venv/bin/python capture_packer.py
../.venv/bin/python validate_mlx_packer.py     # NN layers
../.venv/bin/python validate_mlx_features.py   # featurization + full forward
../.venv/bin/python validate_mlx_geometry.py   # geometry vs OpenFold
../.venv/bin/python validate_mlx_repack.py     # full pipeline coords->atom14
```

## Key wins
- The ported geometry uses only **matmul / gather / add / mul** and stays **rank <= 5**
  (replaced OpenFold's one-hot x frames x sum with a per-atom `gather`), so it maps
  1:1 to Metal/MLX ops.
- Bonus: because it's rank <= 5, the ported geometry **also lowers to Core ML** (the
  original rank-6 version did not) -- so repack geometry is no longer a Core ML blocker either.
- For `repack_everything=True` (the default), side-chain torsions initialize randomly, so
  `atom37_to_torsion_angles` (extracting torsions from input coords) is NOT needed -- the
  geometry port above + aatype remap is the whole geometry half.

## Run
```bash
../.venv/bin/python geometry_port.py            # dump constants
../.venv/bin/python validate_geometry_port.py   # validate vs OpenFold (3 proteins)
```

## Next
Port the NN torsion predictor. It's the same MPNN op family as the (validated,
Core ML-exportable) design model plus a small torsion head + ligand-context
encoders -- so either Core ML export (now that geometry lowers too) or an MLX
reimplementation. Then wire the host-side denoising loop + von Mises mode.
