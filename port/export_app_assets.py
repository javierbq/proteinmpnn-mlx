#!/usr/bin/env python3
"""
Export bundled app assets for MPNNBench:
  inputs/<id>.safetensors   : X[L,4,3]f32, R_idx[L]i32, chain_labels[L]i32, S_native[L]i32
  oracles/<id>_design.safetensors : decoding_order[L]i32, design_top1[L]i32, design_logits[L,21]f32
  oracles/<id>_repack.safetensors : atom14[L,14,3]f32, atom14_mask[L,14]f32, b_factors[L,14]f32
  geometry.safetensors      : RRGDF, GRP, AMASK, LIT (repack geometry constants)
  manifest.json             : [{id, L, n_chains, synthetic, native_seq}]

Design oracle = validated mlx_design.design_greedy (== PyTorch, greedy fixed order).
Repack oracle = mlx_repack_full.repack_full (once available); --skip-repack to defer.

Real proteins parsed via LigandMPNN's pipeline; synthetic entries are a monomer's
backbone tiled with a large translation + chain break per copy (clean O(L^2) sweep).
"""
import os
import sys
import json
import argparse
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from capture_design import build_feature_dict, CKPT                 # noqa: E402
from mlx_design import load_design_weights, design_greedy, ALPHABET  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "ProteinMPNN"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "MPNNBench", "Resources", "app_assets"))

REAL = [
    ("6MRR", f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb"),
    ("5L33", f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb"),
    ("4GYT", f"{REPO}/inputs/PDB_homooligomers/pdbs/4GYT.pdb"),
    ("3HTN", f"{REPO}/inputs/PDB_complexes/pdbs/3HTN.pdb"),
    ("4YOW", f"{REPO}/inputs/PDB_complexes/pdbs/4YOW.pdb"),
    ("6EHB", f"{REPO}/inputs/PDB_homooligomers/pdbs/6EHB.pdb"),
]
# synthetic ceiling: (id, base_pdb, n_copies) — 5L33 monomer is 106 res
SYNTH = [
    ("synth1272", f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb", 12),   # ~1272
    ("synth1590", f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb", 15),   # ~1590
    ("synth2120", f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb", 20),   # ~2120
]


def _np(a):
    return np.asarray(a)


def save_st(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mx.save_safetensors(path, {k: (v if isinstance(v, mx.array) else mx.array(v)) for k, v in d.items()},
                        metadata={"format": "mpnnbench"})


def decoding_order_for(L, seed=0):
    rng = np.random.RandomState(seed)
    randn = np.abs(rng.randn(1, L)).astype(np.float32)
    return np.argsort(randn, axis=-1).astype(np.int32)   # chain_mask all ones


def tile_backbone(pdb, n_copies, sep=140.0):
    """Tile a monomer backbone n_copies times with a per-copy translation + chain break."""
    fd, L0 = build_feature_dict(pdb)
    X0 = _np(fd["X"])[0]                        # [L0,4,3]
    S0 = _np(fd["S"])[0]                        # [L0]
    Xs, Ss, Ridx, Chains = [], [], [], []
    for c in range(n_copies):
        Xs.append(X0 + np.array([sep * c, 0.0, 0.0], np.float32))
        Ss.append(S0)
        Ridx.append(np.arange(L0, dtype=np.int32) + c * (L0 + 100))
        Chains.append(np.full(L0, c, dtype=np.int32))
    X = np.concatenate(Xs, 0)[None]
    S = np.concatenate(Ss, 0)[None]
    R = np.concatenate(Ridx, 0)[None]
    C = np.concatenate(Chains, 0)[None]
    L = X.shape[1]
    mask = np.ones((1, L), np.float32)
    Y = np.zeros((1, L, 25, 3), np.float32)
    Y_m = np.zeros((1, L, 25), np.float32)
    Y_t = np.zeros((1, L, 25), np.int32)
    fd2 = dict(X=X, S=S, mask=mask, R_idx=R, chain_labels=C,
               Y=Y, Y_m=Y_m, Y_t=Y_t, chain_mask=np.ones((1, L), np.float32))
    return fd2, L


def fd_from_pdb(pdb):
    fd, L = build_feature_dict(pdb)
    return {k: _np(fd[k]) for k in ("X", "S", "mask", "R_idx", "chain_labels",
                                    "Y", "Y_m", "Y_t", "chain_mask")}, L


def run_entry(w, eid, fd, L, synthetic, do_repack):
    order = decoding_order_for(L)
    out = design_greedy(
        w, X=mx.array(fd["X"].astype(np.float32)), S_native=mx.array(fd["S"].astype(np.int32)),
        mask=mx.array(fd["mask"].astype(np.float32)),
        R_idx=mx.array(fd["R_idx"].astype(np.float32)),
        chain_labels=mx.array(fd["chain_labels"].astype(np.float32)),
        Y=mx.array(fd["Y"].astype(np.float32)), Y_m=mx.array(fd["Y_m"].astype(np.float32)),
        Y_t=mx.array(fd["Y_t"].astype(np.int32)),
        chain_mask=mx.array(fd["chain_mask"].astype(np.float32)),
        decoding_order=mx.array(order), top_k=32)
    S_des = _np(out["S"])[0].astype(np.int32)
    logits = _np(out["logits"])[0].astype(np.float32)

    save_st(f"{ASSETS}/inputs/{eid}.safetensors", dict(
        X=fd["X"][0].astype(np.float32), R_idx=fd["R_idx"][0].astype(np.int32),
        chain_labels=fd["chain_labels"][0].astype(np.int32), S_native=fd["S"][0].astype(np.int32)))
    save_st(f"{ASSETS}/oracles/{eid}_design.safetensors", dict(
        decoding_order=order[0].astype(np.int32), design_top1=S_des, design_logits=logits))

    rep = ""
    if do_repack:
        from mlx_repack_full import repack_full                      # noqa
        r = repack_full(w_packer, X_bb=mx.array(fd["X"].astype(np.float32)),
                        S=mx.array(S_des[None]), mask=mx.array(fd["mask"].astype(np.float32)),
                        R_idx=mx.array(fd["R_idx"].astype(np.float32)),
                        chain_labels=mx.array(fd["chain_labels"].astype(np.float32)))
        save_st(f"{ASSETS}/oracles/{eid}_repack.safetensors", dict(
            atom14=_np(r["atom14"])[0].astype(np.float32),
            atom14_mask=_np(r["atom14_mask"])[0].astype(np.float32),
            b_factors=_np(r["b_factors"])[0].astype(np.float32)))
        rep = "  +repack"

    n_ch = int(len(np.unique(fd["chain_labels"][0])))
    seq = "".join(ALPHABET[i] for i in S_des.tolist())
    print(f"[{eid}] L={L} chains={n_ch} synthetic={synthetic}{rep}  seq[:40]={seq[:40]}")
    return dict(id=eid, L=int(L), n_chains=n_ch, synthetic=synthetic, native_seq=seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-repack", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated ids to run")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    global w_packer
    w = load_design_weights(CKPT)
    w_packer = None
    if not args.skip_repack:
        from mlx_packer import load_weights
        PACKER = os.path.abspath(os.path.join(HERE, "..", "..", "LigandMPNN",
                                              "model_params", "ligandmpnn_sc_v_32_002_16.pt"))
        w_packer = load_weights(PACKER)

    # geometry constants
    gc = np.load(f"{HERE}/geometry_constants.npz")
    save_st(f"{ASSETS}/geometry.safetensors", dict(
        RRGDF=gc["restype_rigid_group_default_frame"].astype(np.float32),
        GRP=gc["restype_atom14_to_rigid_group"].astype(np.int32),
        AMASK=gc["restype_atom14_mask"].astype(np.float32),
        LIT=gc["restype_atom14_rigid_group_positions"].astype(np.float32)))

    manifest = []
    for eid, pdb in REAL:
        if only and eid not in only:
            continue
        fd, L = fd_from_pdb(pdb)
        manifest.append(run_entry(w, eid, fd, L, False, not args.skip_repack))
    for eid, pdb, ncop in SYNTH:
        if only and eid not in only:
            continue
        fd, L = tile_backbone(pdb, ncop)
        manifest.append(run_entry(w, eid, fd, L, True, not args.skip_repack))

    # merge with any existing manifest (so partial runs accumulate)
    mpath = f"{ASSETS}/manifest.json"
    existing = {}
    if os.path.exists(mpath):
        existing = {m["id"]: m for m in json.load(open(mpath))}
    for m in manifest:
        existing[m["id"]] = m
    order_ids = [e[0] for e in REAL] + [e[0] for e in SYNTH]
    merged = [existing[i] for i in order_ids if i in existing]
    json.dump(merged, open(mpath, "w"), indent=2)
    print(f"\nwrote manifest with {len(merged)} entries -> {mpath}")


if __name__ == "__main__":
    main()
