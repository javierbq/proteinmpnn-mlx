#!/usr/bin/env python3
"""Validate the MLX featurization port (+ full coords->torsions chain) vs PyTorch fixtures."""
import os
import numpy as np
import mlx.core as mx
import mlx_packer as M
import mlx_features as F

HERE = os.path.dirname(os.path.abspath(__file__))
LIG = "/Users/jcastellanos/repos/LigandMPNN"
cap = np.load(f"{HERE}/packer_capture.npz")
w = M.load_weights(f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt")
pt = mx.array(cap["periodic_table_features"])

def A(n, dtype=None):
    x = cap[n]
    return mx.array(x.astype(dtype) if dtype else x)

def cmp(name, got, ref, tol=2e-4, rel=False):
    got = np.array(got); ref = np.asarray(ref)
    if got.shape != ref.shape:
        print(f"[feat] {name:18s} SHAPE MISMATCH got{got.shape} ref{ref.shape}  FAIL"); return False
    if rel:
        v = float((np.abs(got - ref) / (np.abs(ref) + 1e-6)).max()); tol = 1e-3
    else:
        v = float(np.abs(got - ref).max())
    print(f"[feat] {name:18s} shape={tuple(got.shape)}  {'max_rel' if rel else 'max|diff|'}={v:.3e}  {'PASS' if v < tol else 'FAIL'}")
    return v < tol

def main():
    ok = True
    # ---- features_encode ----
    V, E, E_idx, Yn, Ye, Ec, Ym = F.features_encode(
        w, pt, A("enc_in_S", np.int32), A("enc_in_X"), A("enc_in_Y"), A("enc_in_Y_m"),
        A("enc_in_Y_t", np.int32), A("enc_in_mask"), A("enc_in_R_idx", np.float32),
        A("enc_in_chain_labels", np.float32))
    eidx_match = int((np.array(E_idx) == cap["E_idx"]).all())
    print(f"[feat] E_idx exact match vs PyTorch topk: {bool(eidx_match)}")
    ok &= bool(eidx_match)
    ok &= cmp("enc V", V, cap["V"])
    ok &= cmp("enc E", E, cap["E"])
    ok &= cmp("enc Y_nodes", Yn, cap["Y_nodes"])
    ok &= cmp("enc Y_edges", Ye, cap["Y_edges"])
    ok &= cmp("enc E_context", Ec, cap["E_context"])

    # ---- features_decode ----
    dV, dF = F.features_decode(
        w, A("dec_in_S", np.int32), A("dec_in_X"), A("dec_in_X_m"), A("dec_in_mask"),
        A("dec_in_E_idx", np.int32), A("dec_in_Y"), A("dec_in_Y_m"), A("dec_in_Y_t", np.int32))
    ok &= cmp("dec V", dV, cap["dec_V"])
    ok &= cmp("dec F", dF, cap["dec_F"])

    # ---- FULL CHAIN: coords -> featurize -> encode -> decode -> torsions ----
    h_V, h_E = M.encode(w, V, E, E_idx, Yn, Ye, Ec, Ym, A("enc_in_mask"))
    mean, conc, mix = M.decode(w, dV, dF, h_V, h_E, E_idx, A("dec_in_mask"))
    print("-" * 64)
    ok &= cmp("FULL mean", mean, cap["mean"])
    ok &= cmp("FULL concentration", conc, cap["concentration"], rel=True)
    ok &= cmp("FULL mix_logits", mix, cap["mix_logits"])
    print("-" * 64)
    print("[feat] FULL PACKER FORWARD (coords->torsions) MATCHES PyTorch" if ok else "[feat] MISMATCH")

if __name__ == "__main__":
    main()
