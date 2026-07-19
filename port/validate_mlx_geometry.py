#!/usr/bin/env python3
"""Validate the MLX geometry transcription against OpenFold (full chain, 3 proteins)."""
import os, numpy as np, mlx.core as mx
from validate_geometry_port import capture, PDBS   # reuse torch/OpenFold capture
import mlx_geometry as G

HERE = os.path.dirname(os.path.abspath(__file__))
c = np.load(f"{HERE}/geometry_constants.npz")
RRGDF = mx.array(c["restype_rigid_group_default_frame"])
GRP = mx.array(c["restype_atom14_to_rigid_group"].astype(np.int32))
AMASK = mx.array(c["restype_atom14_mask"])
LIT = mx.array(c["restype_atom14_rigid_group_positions"])

def main():
    print(f"{'protein':8s} | {'frame R':>10s}{'frame t':>10s} | {'full coords->atom14 (A)':>24s}  result")
    print("-" * 62)
    allok = True
    for name, pdb in PDBS.items():
        cap = capture(pdb)
        bb = cap["bb"].numpy(); alpha = mx.array(cap["alpha"].numpy())
        aatype = mx.array(cap["aatype"].numpy().astype(np.int32)); ref = cap["ref"].numpy()
        NCAC = cap["NCAC"].numpy()
        # (1) backbone frame from coords vs OpenFold captured frame
        pR, pt = G.make_backbone_frame(mx.array(NCAC[:, :, 0]), mx.array(NCAC[:, :, 1]), mx.array(NCAC[:, :, 2]))
        dR = float(np.abs(np.array(pR) - bb[..., :3, :3]).max())
        dt = float(np.abs(np.array(pt) - bb[..., :3, 3]).max())
        # (2) full chain coords -> atom14
        gR, gt = G.torsion_angles_to_frames(pR, pt, alpha, aatype, RRGDF)
        pos = G.frames_to_atom14(gR, gt, aatype, GRP, AMASK, LIT)
        dfull = float(np.abs(np.array(pos) - ref).max())
        ok = max(dR, dt, dfull) < 1e-4
        allok &= ok
        print(f"{name:8s} | {dR:10.2e}{dt:10.2e} | {dfull:24.2e}  {'PASS' if ok else 'FAIL'}")
    print("-" * 62)
    print("[mlxgeom] MLX GEOMETRY MATCHES OpenFold" if allok else "[mlxgeom] MISMATCH")

if __name__ == "__main__":
    main()
