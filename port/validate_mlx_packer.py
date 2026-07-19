#!/usr/bin/env python3
"""Validate the MLX Packer NN (encode + decode) against captured PyTorch fixtures."""
import os
import numpy as np
import mlx.core as mx
import mlx_packer as M

HERE = os.path.dirname(os.path.abspath(__file__))
LIG = "/Users/jcastellanos/repos/LigandMPNN"
cap = np.load(f"{HERE}/packer_capture.npz")
w = M.load_weights(f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt")

def a(name):  # fixture -> mx.array
    return mx.array(cap[name])

def cmp(name, got, ref, rel=False):
    got = np.array(got); ref = np.asarray(ref)
    md = float(np.abs(got - ref).max())
    if rel:
        rd = float((np.abs(got - ref) / (np.abs(ref) + 1e-6)).max())
        print(f"[mlx] {name:16s} shape={tuple(got.shape)}  max_rel_diff={rd:.3e}  {'PASS' if rd < 1e-3 else 'FAIL'}")
        return rd < 1e-3
    print(f"[mlx] {name:16s} shape={tuple(got.shape)}  max|diff|={md:.3e}  {'PASS' if md < 2e-4 else 'FAIL'}")
    return md < 2e-4

def main():
    E_idx = mx.array(cap["E_idx"].astype(np.int32))
    ok = True

    # --- encode ---
    h_V, h_E = M.encode(w, a("V"), a("E"), E_idx, a("Y_nodes"), a("Y_edges"),
                        a("E_context"), a("Y_m"), a("mask"))
    ok &= cmp("encode h_V", h_V, cap["h_V"])
    ok &= cmp("encode h_E", h_E, cap["h_E"])

    # --- decode (use captured encode + featurize-decode outputs as inputs) ---
    mean, conc, mix = M.decode(w, a("dec_V"), a("dec_F"), mx.array(cap["h_V"]),
                               mx.array(cap["h_E"]), E_idx, a("mask"))
    ok &= cmp("decode mean", mean, cap["mean"])
    ok &= cmp("decode concentration", conc, cap["concentration"], rel=True)  # ranges to ~600
    ok &= cmp("decode mix_logits", mix, cap["mix_logits"])

    print("-" * 60)
    print("[mlx] PACKER NN (encode+decode) MATCHES PyTorch" if ok else "[mlx] MISMATCH")

if __name__ == "__main__":
    main()
