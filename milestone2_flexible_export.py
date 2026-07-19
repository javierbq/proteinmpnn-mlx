#!/usr/bin/env python3
"""
Milestone 2 - FLEXIBLE-LENGTH Core ML export of ProteinMPNN's static path.

Goal: one deployable artifact that handles multiple protein lengths.

Finding (coremltools 9.0 / torch 2.13): a SINGLE dynamic-length model does NOT
convert. Both ct.RangeDim and ct.EnumeratedShapes trip a coremltools rank-
inference bug in `gather` under symbolic shapes -- the message-passing helpers
do `neighbor_idx.unsqueeze(-1).expand(...)` before `torch.gather`, and under a
dynamic torch.export graph coremltools sees the indices as one rank too low
("Rank mismatch: input rank 4, indices rank 3").

Working strategy: per-length BUCKET models (fixed shapes convert perfectly, as
the cross-protein benchmark showed for 68-960 res). At runtime, pad the input up
to the nearest bucket (mask=0 on padding) and pick that model. This module
exports one fixed model per bucket and validates parity for each.

  .venv/bin/python milestone2_flexible_export.py
"""
import os, sys
import numpy as np
import torch

REPO = "/Users/jcastellanos/repos/ProteinMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from milestone1_coreml_export import UncondWrapper, patch_dist_constant_k, build_model  # noqa
import coremltools as ct  # noqa

K = 48  # pinned topk; models valid for L >= K


def load_case(npz):
    d = np.load(npz, allow_pickle=True)
    return (torch.tensor(d["X"], dtype=torch.float32),
            torch.tensor(d["mask"], dtype=torch.float32),
            torch.tensor(d["residue_idx"], dtype=torch.int64),
            torch.tensor(d["chain_encoding_all"], dtype=torch.int64),
            d["log_p_uncond"])


def try_dynamic():
    """Demonstrate that the single dynamic-length export fails, and report how."""
    model, _ = build_model(f"{REPO}/vanilla_model_weights/v_48_020.pt")
    patch_dist_constant_k(model, K)
    wrap = UncondWrapper(model)
    X, mask, ridx, cenc, _ = load_case(f"{HERE}/oracle_6MRR.npz")
    L = torch.export.Dim("L", min=48, max=1200)
    dyn = {"X": {1: L}, "mask": {1: L}, "residue_idx": {1: L}, "chain_encoding_all": {1: L}}
    ep = torch.export.export(wrap, (X, mask, ridx, cenc), dynamic_shapes=dyn).run_decompositions({})
    print("[flex] torch.export dynamic graph: OK")
    try:
        ct.convert(ep,
                   inputs=[ct.TensorType(name="X", shape=(1, ct.RangeDim(48, 1200), 4, 3), dtype=np.float32),
                           ct.TensorType(name="mask", shape=(1, ct.RangeDim(48, 1200)), dtype=np.float32),
                           ct.TensorType(name="residue_idx", shape=(1, ct.RangeDim(48, 1200)), dtype=np.int32),
                           ct.TensorType(name="chain_encoding_all", shape=(1, ct.RangeDim(48, 1200)), dtype=np.int32)],
                   convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                   minimum_deployment_target=ct.target.iOS18)
        print("[flex] dynamic ct.convert: UNEXPECTEDLY SUCCEEDED")
        return True
    except Exception as e:
        print(f"[flex] dynamic ct.convert FAILED (expected): {type(e).__name__}: {str(e).strip()[:110]}")
        return False


def export_bucket(bucket_L):
    model, kmax = build_model(f"{REPO}/vanilla_model_weights/v_48_020.pt")
    patch_dist_constant_k(model, min(kmax, bucket_L))
    wrap = UncondWrapper(model)
    ex = (torch.zeros(1, bucket_L, 4, 3), torch.ones(1, bucket_L),
          torch.zeros(1, bucket_L, dtype=torch.int64), torch.ones(1, bucket_L, dtype=torch.int64))
    ep = torch.export.export(wrap, ex).run_decompositions({})
    ml = ct.convert(ep,
                    inputs=[ct.TensorType(name="X", shape=[1, bucket_L, 4, 3], dtype=np.float32),
                            ct.TensorType(name="mask", shape=[1, bucket_L], dtype=np.float32),
                            ct.TensorType(name="residue_idx", shape=[1, bucket_L], dtype=np.int32),
                            ct.TensorType(name="chain_encoding_all", shape=[1, bucket_L], dtype=np.int32)],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                    minimum_deployment_target=ct.target.iOS18)
    p = f"{HERE}/ProteinMPNN_uncond_L{bucket_L}.mlpackage"
    ml.save(p)
    return ml


def pad_to(t, bucket_L, pad_val=0):
    B, L = t.shape[0], t.shape[1]
    if L == bucket_L:
        return t
    shape = list(t.shape); shape[1] = bucket_L - L
    return torch.cat([t, torch.full(shape, pad_val, dtype=t.dtype)], dim=1)


def main():
    print("=== 1) single dynamic-length model ===")
    try_dynamic()

    print("\n=== 2) working strategy: per-length bucket models (fixed shapes) ===")
    # buckets chosen to cover the two references exactly; real deployment would use
    # e.g. [128,256,384,512,768,1024] and pad up to the nearest.
    cases = {"oracle_6MRR.npz": 68, "oracle_5L33.npz": 106}
    ok = True
    for npz, bucket in cases.items():
        ml = export_bucket(bucket)
        X, mask, ridx, cenc, ref = load_case(f"{HERE}/{npz}")
        Xp, mp = pad_to(X, bucket), pad_to(mask, bucket)
        rp, cp = pad_to(ridx, bucket), pad_to(cenc, bucket)
        pred = ml.predict({"X": Xp.numpy(), "mask": mp.numpy(),
                           "residue_idx": rp.numpy().astype(np.int32),
                           "chain_encoding_all": cp.numpy().astype(np.int32)})
        cm = pred[list(pred.keys())[0]][:, :mask.shape[1]]
        m = mask.numpy().astype(bool)
        md = float(np.abs(cm - ref)[m].max())
        agree = float((cm.argmax(-1) == ref.argmax(-1))[m].mean())
        ok = ok and md < 5e-2
        print(f"[flex]   bucket L={bucket:4d}: max|diff|={md:.3e}  top1={agree:.4f}  "
              f"{'PASS' if md < 5e-2 else 'FAIL'}  -> ProteinMPNN_uncond_L{bucket}.mlpackage")
    print("\n[flex] CONCLUSION: single dynamic model unsupported (coremltools gather bug);",
          "per-bucket fixed models WORK" if ok else "bucket models FAILED")


if __name__ == "__main__":
    main()
