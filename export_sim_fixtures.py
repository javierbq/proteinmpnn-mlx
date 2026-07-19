#!/usr/bin/env python3
"""Dump oracle inputs + reference output as raw little-endian binaries for the
Swift/Core ML simulator test, and print the .mlpackage I/O signature."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "oracle_6MRR.npz"), allow_pickle=True)
fx = os.path.join(HERE, "sim_fixtures"); os.makedirs(fx, exist_ok=True)

X = d["X"].astype("<f4")
mask = d["mask"].astype("<f4")
residue_idx = d["residue_idx"].astype("<i4")
chain_enc = d["chain_encoding_all"].astype("<i4")
ref = d["log_p_uncond"].astype("<f4")

X.tofile(os.path.join(fx, "X.bin"))
mask.tofile(os.path.join(fx, "mask.bin"))
residue_idx.tofile(os.path.join(fx, "residue_idx.bin"))
chain_enc.tofile(os.path.join(fx, "chain_encoding_all.bin"))
ref.tofile(os.path.join(fx, "ref_logp.bin"))
meta = {"B": int(mask.shape[0]), "L": int(mask.shape[1]), "A": 21}
json.dump(meta, open(os.path.join(fx, "shapes.json"), "w"))
print(f"[fixtures] wrote {fx}  B={meta['B']} L={meta['L']}")

# print CoreML I/O signature so the Swift side uses correct names
try:
    import coremltools as ct
    m = ct.models.MLModel(os.path.join(HERE, "ProteinMPNN_uncond.mlpackage"))
    spec = m.get_spec()
    print("[fixtures] inputs :", [(f.name, f.type.WhichOneof("Type")) for f in spec.description.input])
    print("[fixtures] outputs:", [(f.name, f.type.WhichOneof("Type")) for f in spec.description.output])
except Exception as e:
    print("[fixtures] could not read mlpackage spec:", e)
