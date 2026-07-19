#!/usr/bin/env python3
"""
End-to-end benchmark: sequence DESIGN (LigandMPNN) -> side-chain REPACK
(LigandMPNN side-chain packer, OpenFold torsion/frame geometry), across a range
of proteins. Assesses stability/robustness + per-stage timing of the full
design+repack pipeline (the realistic on-device workflow).

  .venv/bin/python benchmark_design_repack.py
"""
import os, sys, time, json, copy
import numpy as np
import torch
import logging

LIG = "/Users/jcastellanos/repos/LigandMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
sys.path.insert(0, LIG)
from data_utils import parse_PDB, featurize, get_score      # noqa
from model_utils import ProteinMPNN                          # noqa
from sc_utils import Packer, pack_side_chains                # noqa
logging.getLogger("prody").setLevel(logging.ERROR)

DEVICE = torch.device("cpu")
DESIGN_CKPT = f"{LIG}/model_params/ligandmpnn_v_32_010_25.pt"
SC_CKPT = f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt"
PDBS = [
    f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb",
    f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb",
    f"{REPO}/inputs/PDB_homooligomers/pdbs/4GYT.pdb",
    f"{REPO}/inputs/PDB_complexes/pdbs/3HTN.pdb",
    f"{REPO}/inputs/PDB_complexes/pdbs/4YOW.pdb",
    f"{REPO}/inputs/PDB_homooligomers/pdbs/6EHB.pdb",
]
TEMP, SC_STEPS, SC_SAMPLES = 0.1, 3, 16


def load_design_model():
    ck = torch.load(DESIGN_CKPT, map_location=DEVICE)
    m = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                    num_encoder_layers=3, num_decoder_layers=3,
                    k_neighbors=ck["num_edges"], device=DEVICE,
                    atom_context_num=ck["atom_context_num"], model_type="ligand_mpnn",
                    ligand_mpnn_use_side_chain_context=0)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m, ck["atom_context_num"]


def load_sc_model():
    ck = torch.load(SC_CKPT, map_location=DEVICE)
    m = Packer(node_features=128, edge_features=128, num_positional_embeddings=16,
               num_chain_embeddings=16, num_rbf=16, hidden_dim=128,
               num_encoder_layers=3, num_decoder_layers=3, atom_context_num=16,
               lower_bound=0.0, upper_bound=20.0, top_k=32, dropout=0.0,
               augment_eps=0.0, atom37_order=False, device=DEVICE, num_mix=3)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m


def run_one(path, model, atom_context_num, model_sc):
    dl_name = os.path.basename(path).split(".")[0]
    pd, *_ = parse_PDB(path, device=DEVICE, chains=[], parse_all_atoms=True)
    pd["chain_mask"] = torch.ones_like(pd["mask"])
    n_chains = int(len(set(pd["chain_labels"].tolist())))

    fd = featurize(pd, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=atom_context_num, model_type="ligand_mpnn")
    B, L = fd["X"].shape[0], fd["X"].shape[1]
    fd["batch_size"] = 1
    fd["temperature"] = TEMP
    fd["bias"] = torch.zeros([1, L, 21], device=DEVICE)
    fd["symmetry_residues"] = [[]]
    fd["symmetry_weights"] = [[]]

    # --- DESIGN ---
    torch.manual_seed(111)
    t0 = time.time()
    with torch.no_grad():
        fd["randn"] = torch.randn([1, fd["mask"].shape[1]], device=DEVICE)
        out = model.sample(fd)
    design_ms = (time.time() - t0) * 1000.0
    S_des = out["S"]
    score, _ = get_score(S_des, out["log_probs"], fd["mask"] * fd["chain_mask"])
    design_conf = float(torch.exp(-score).mean())   # ~seq confidence

    # --- REPACK (OpenFold geometry denoiser) on the DESIGNED sequence ---
    sc_fd = copy.deepcopy(fd)
    sc_fd["S"] = S_des.long()
    sc_fd["chain_mask"] = torch.ones_like(fd["mask"])
    t0 = time.time()
    with torch.no_grad():
        sc = pack_side_chains(sc_fd, model_sc, SC_STEPS, SC_SAMPLES, repack_everything=True)
    repack_ms = (time.time() - t0) * 1000.0

    X14, X14m = sc["X"], sc["X_m"]
    finite = torch.isfinite(X14).all(-1).float()
    placed = float((finite * X14m).sum())
    expected = float(X14m.sum())
    completeness = placed / max(expected, 1.0)
    nan_free = bool(torch.isfinite(X14[X14m.bool()]).all())
    mean_conf = float(sc["b_factors"][X14m.bool()].mean())  # von Mises log-prob confidence

    return dict(name=dl_name, L=int(L), chains=n_chains,
                design_ms=design_ms, repack_ms=repack_ms, total_ms=design_ms + repack_ms,
                design_conf=design_conf, sc_atoms=int(expected),
                completeness=completeness, nan_free=nan_free, mean_sc_conf=mean_conf)


def main():
    model, acn = load_design_model()
    model_sc = load_sc_model()
    print(f"[bench] design={sum(p.numel() for p in model.parameters()):,} params  "
          f"packer={sum(p.numel() for p in model_sc.parameters()):,} params  "
          f"(steps={SC_STEPS}, samples/step={SC_SAMPLES})")
    rows = []
    for p in PDBS:
        try:
            r = run_one(p, model, acn, model_sc)
            rows.append(r)
            print(f"[bench] {r['name']:6s} L={r['L']:4d} ch={r['chains']} | "
                  f"design {r['design_ms']:7.1f} ms | repack {r['repack_ms']:8.1f} ms | "
                  f"sc_atoms={r['sc_atoms']:5d} complete={r['completeness']:.4f} "
                  f"nan_free={r['nan_free']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[bench] {os.path.basename(p)}: ERROR {type(e).__name__}: {str(e)[:140]}")
            rows.append({"name": os.path.basename(p), "error": str(e)[:200]})
    json.dump(rows, open(f"{HERE}/benchmark_design_repack.json", "w"), indent=2)
    print(f"\n[bench] saved -> {HERE}/benchmark_design_repack.json")


if __name__ == "__main__":
    main()
