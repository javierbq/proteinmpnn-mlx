#!/usr/bin/env python3
"""
Stability benchmark: export ProteinMPNN's static path (unconditional_probs) to
Core ML for a range of proteins (varying length, chain count, complexes/oligomers)
and measure numerical parity vs the PyTorch fp32 reference, plus latency.

Assesses whether the export stays correct & robust across inputs -- and quantifies
the fp16 kNN-graph instability across sizes.

  .venv/bin/python benchmark_proteins.py
"""
import os, sys, time, json, glob
import numpy as np
import torch
import logging

REPO = "/Users/jcastellanos/repos/ProteinMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from protein_mpnn_utils import parse_PDB, tied_featurize, StructureDatasetPDB  # noqa
from milestone1_coreml_export import UncondWrapper, patch_dist_constant_k, build_model  # noqa
import coremltools as ct  # noqa
for n in ("coremltools", "coremltools.converters"):
    logging.getLogger(n).setLevel(logging.ERROR)

DEVICE = torch.device("cpu")
PDBS = [
    f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb",
    f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb",
    f"{REPO}/inputs/PDB_homooligomers/pdbs/4GYT.pdb",
    f"{REPO}/inputs/PDB_complexes/pdbs/3HTN.pdb",
    f"{REPO}/inputs/PDB_complexes/pdbs/4YOW.pdb",
    f"{REPO}/inputs/PDB_homooligomers/pdbs/6EHB.pdb",
]


def featurize(path):
    dl = parse_PDB(path, ca_only=False)
    ds = StructureDatasetPDB(dl, truncate=None, max_length=20000)
    name = dl[0]["name"]
    chains = [k[-1:] for k in dl[0] if k[:9] == "seq_chain"]
    cid = {name: (chains, [])}
    t = tied_featurize([ds[0]], DEVICE, cid, None, None, None, None, None, ca_only=False)
    X, S, mask, _, chain_M, chain_enc = t[0], t[1], t[2], t[3], t[4], t[5]
    residue_idx = t[12]
    return name, len(chains), X, S, mask, residue_idx, chain_enc


def convert_and_eval(model, ex_inputs, shapes, ref, mask_np, precision, n_time=20):
    prec = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    B, L = shapes
    inputs = [
        ct.TensorType(name="X", shape=[B, L, 4, 3], dtype=np.float32),
        ct.TensorType(name="mask", shape=[B, L], dtype=np.float32),
        ct.TensorType(name="residue_idx", shape=[B, L], dtype=np.int32),
        ct.TensorType(name="chain_encoding_all", shape=[B, L], dtype=np.int32),
    ]
    ep = torch.export.export(model, ex_inputs).run_decompositions({})
    ml = ct.convert(ep, inputs=inputs, convert_to="mlprogram",
                    compute_precision=prec, minimum_deployment_target=ct.target.iOS18)
    feed = {"X": ex_inputs[0].numpy(), "mask": ex_inputs[1].numpy(),
            "residue_idx": ex_inputs[2].numpy().astype(np.int32),
            "chain_encoding_all": ex_inputs[3].numpy().astype(np.int32)}
    out = ml.predict(feed)
    cm = out[list(out.keys())[0]]
    m = mask_np.astype(bool)
    md = float(np.abs(cm - ref)[m].max())
    agree = float((cm.argmax(-1) == ref.argmax(-1))[m].mean())
    t0 = time.time()
    for _ in range(n_time):
        ml.predict(feed)
    lat = (time.time() - t0) / n_time * 1000.0
    return md, agree, lat


def main():
    rows = []
    for path in PDBS:
        name, nch, X, S, mask, residue_idx, chain_enc = featurize(path)
        B, L = mask.shape
        with torch.no_grad():
            ref = build_ref(X, mask, residue_idx, chain_enc)
        ex = (X.float(), mask.float(), residue_idx.long(), chain_enc.long())
        # fresh model + constant-k patch per protein
        model, kmax = build_model(f"{REPO}/vanilla_model_weights/v_48_020.pt")
        patch_dist_constant_k(model, min(kmax, L))
        wrap = UncondWrapper(model)
        mnp = mask.numpy()
        r = {"name": name, "L": int(L), "chains": nch}
        for prec in ("fp32", "fp16"):
            try:
                md, agree, lat = convert_and_eval(wrap, ex, (B, L), ref, mnp, prec)
                r[prec] = dict(max_diff=md, top1=agree, ms=lat)
                print(f"[bench] {name:6s} L={L:4d} ch={nch} {prec}: "
                      f"max|diff|={md:.2e} top1={agree:.4f} {lat:7.1f} ms")
            except Exception as e:
                r[prec] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                print(f"[bench] {name:6s} L={L:4d} {prec}: ERROR {r[prec]['error']}")
        rows.append(r)

    # summary table
    print("\n=== STABILITY SUMMARY (unconditional_probs Core ML export) ===")
    print(f"{'protein':8s}{'L':>6s}{'ch':>4s} | {'fp32 max|diff|':>15s}{'fp32 top1':>11s}{'fp32 ms':>9s} | "
          f"{'fp16 max|diff|':>15s}{'fp16 top1':>11s}")
    for r in rows:
        f32 = r.get("fp32", {}); f16 = r.get("fp16", {})
        print(f"{r['name']:8s}{r['L']:6d}{r['chains']:4d} | "
              f"{f32.get('max_diff', float('nan')):15.2e}{f32.get('top1', float('nan')):11.4f}{f32.get('ms', float('nan')):9.1f} | "
              f"{f16.get('max_diff', float('nan')):15.2e}{f16.get('top1', float('nan')):11.4f}")
    json.dump(rows, open(f"{HERE}/benchmark_results.json", "w"), indent=2)
    print(f"\n[bench] saved -> {HERE}/benchmark_results.json")


def build_ref(X, mask, residue_idx, chain_enc):
    # fp32 PyTorch reference via a clean model instance
    model, kmax = build_model(f"{REPO}/vanilla_model_weights/v_48_020.pt")
    return model.unconditional_probs(X, mask, residue_idx, chain_enc).numpy()


if __name__ == "__main__":
    main()
