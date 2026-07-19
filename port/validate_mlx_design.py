#!/usr/bin/env python3
"""Validate mlx_design.py against the PyTorch oracle captures (capture_design.py).

Stage-by-stage parity:  featurization (E_idx exact, E), encode (h_V/h_E),
autoregressive greedy decode (logits, S top-1). RNG-free so it must match closely.
"""
import os
import sys
import glob
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
from mlx_design import load_design_weights, design_greedy, ALPHABET   # noqa: E402

CKPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "LigandMPNN", "model_params", "ligandmpnn_v_32_010_25.pt"))


def md(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    w = load_design_weights(CKPT)
    caps = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "design_capture_*.npz")))
    assert caps, "no design_capture_*.npz found; run capture_design.py first"

    overall_ok = True
    for cap_path in caps:
        cid = os.path.basename(cap_path)[len("design_capture_"):-len(".npz")]
        d = np.load(cap_path, allow_pickle=True)
        assert int(np.asarray(d["Y_t"]).sum()) == 0, "expected protein-only (Y_t all zero)"

        out = design_greedy(
            w,
            X=mx.array(d["X"]), S_native=mx.array(d["S_native"]),
            mask=mx.array(d["mask"]), R_idx=mx.array(d["R_idx"].astype(np.float32)),
            chain_labels=mx.array(d["chain_labels"].astype(np.float32)),
            Y=mx.array(d["Y"]), Y_m=mx.array(d["Y_m"]), Y_t=mx.array(d["Y_t"].astype(np.int32)),
            chain_mask=mx.array(d["chain_mask"]),
            decoding_order=mx.array(d["decoding_order"]),
            top_k=32,
        )

        eidx_exact = bool(np.array_equal(np.asarray(out["E_idx"]), d["E_idx"]))
        dE = md(out["E"], d["E"])
        dhV = md(out["h_V"], d["h_V"])
        dhE = md(out["h_E"], d["h_E"])
        dlogit = md(out["logits"], d["logits"])
        S_mlx = np.asarray(out["S"])[0]
        S_ref = d["S"][0]
        top1 = float((S_mlx == S_ref).mean()) * 100.0
        seq_mlx = "".join(ALPHABET[i] for i in S_mlx.tolist())

        ok = eidx_exact and dE < 1e-4 and dhV < 1e-4 and dhE < 1e-4 and dlogit < 1e-3 and top1 == 100.0
        overall_ok &= ok
        L = S_ref.shape[0]
        print(f"[{cid}] L={L}  E_idx_exact={eidx_exact}  |dE|={dE:.2e}  "
              f"|dh_V|={dhV:.2e}  |dh_E|={dhE:.2e}  |dlogits|={dlogit:.2e}  "
              f"top1={top1:.1f}%  {'PASS' if ok else 'FAIL'}")
        if top1 != 100.0:
            print(f"    mlx : {seq_mlx[:70]}")
            print(f"    ref : {str(d['seq'])[:70]}")

    print("\n" + ("ALL PASS" if overall_ok else "FAILURES PRESENT"))
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
