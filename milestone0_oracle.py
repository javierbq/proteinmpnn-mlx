#!/usr/bin/env python3
"""
Milestone 0 - Reference oracle for ProteinMPNN (sequence design).

Runs stock ProteinMPNN on a test PDB with a fixed seed and dumps:
  - unconditional_probs log_p [L,21]        (the clean static path -> Core ML MVP)
  - forward() log_probs [L,21] + native score
  - the exact featurizer tensors (X, S, mask, residue_idx, chain_encoding_all, E_idx)
  - one sampled (designed) sequence + its score

Everything is saved to oracle_<name>.npz for later numerical-parity checks
against the Core ML export (milestone1) and the MLX-Swift port.

Usage:
  .venv/bin/python milestone0_oracle.py \
      --pdb /Users/jcastellanos/repos/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb \
      --weights /Users/jcastellanos/repos/ProteinMPNN/vanilla_model_weights/v_48_020.pt
"""
import argparse, os, sys
import numpy as np
import torch

REPO = "/Users/jcastellanos/repos/ProteinMPNN"
sys.path.insert(0, REPO)
from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN, parse_PDB, tied_featurize, StructureDatasetPDB,
    _scores, _S_to_seq,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb")
    ap.add_argument("--weights", default=f"{REPO}/vanilla_model_weights/v_48_020.pt")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")  # match the deployment target (no CUDA on iOS)

    # ---- Build model (augment_eps=0: NO inference-time coordinate noise) ----
    ckpt = torch.load(args.weights, map_location=device)
    hidden_dim, num_layers = 128, 3
    model = ProteinMPNN(
        num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
        hidden_dim=hidden_dim, num_encoder_layers=num_layers,
        num_decoder_layers=num_layers, augment_eps=0.0,
        k_neighbors=ckpt["num_edges"], ca_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[oracle] model loaded: k_neighbors={ckpt['num_edges']}, "
          f"noise_level={ckpt.get('noise_level')}, params="
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ---- Featurize the PDB (single chain, design everything) ----
    pdb_dict_list = parse_PDB(args.pdb, ca_only=False)
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)
    protein = dataset[0]
    name = pdb_dict_list[0]["name"]
    all_chains = [k[-1:] for k in pdb_dict_list[0] if k[:9] == "seq_chain"]
    chain_id_dict = {name: (all_chains, [])}  # (designed, fixed)

    batch = [protein]
    (X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos,
     omit_AA_mask, residue_idx, dihedral_mask, tied_pos, pssm_coef, pssm_bias,
     pssm_log_odds_all, bias_by_res_all, tied_beta) = tied_featurize(
        batch, device, chain_id_dict, None, None, None, None, None, ca_only=False)
    L = int(mask.sum().item())
    print(f"[oracle] {name}: chains={all_chains}, L(total)={X.shape[1]}, L(resolved)={L}")

    # ---- Grab the graph features directly (E_idx is the kNN graph) ----
    with torch.no_grad():
        E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)

        # (1) unconditional_probs -- the clean static path for the Core ML MVP
        log_p_uncond = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)

        # (2) forward() teacher-forced log_probs + native score
        randn = torch.randn(chain_M.shape, device=device)
        log_probs_fwd = model(X, S, mask, chain_M * chain_M_pos, residue_idx,
                              chain_encoding_all, randn)
        native_score = _scores(S, log_probs_fwd, mask * chain_M * chain_M_pos)

        # (3) one design sample (fixed seed) at low temperature
        omit_AAs_np = np.array([aa in "X" for aa in ALPHABET]).astype(np.float32)
        bias_AAs_np = np.zeros(len(ALPHABET), dtype=np.float32)
        randn_s = torch.randn(chain_M.shape, device=device)
        sd = model.sample(
            X, randn_s, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
            temperature=args.temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
            chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
            pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
            pssm_log_odds_mask=(pssm_log_odds_all > 0.0).float(),
            pssm_bias_flag=False, bias_by_res=bias_by_res_all)
        S_sample = sd["S"]
        seq = _S_to_seq(S_sample[0], chain_M[0])
        native_seq = _S_to_seq(S[0], chain_M[0])

    print(f"[oracle] native score (mean -log p): {float(native_score.mean()):.4f}")
    print(f"[oracle] native seq : {native_seq}")
    print(f"[oracle] design seq : {seq}")
    recov = np.mean([a == b for a, b in zip(seq, native_seq)])
    print(f"[oracle] seq recovery vs native: {recov:.3f}")

    out = args.out or f"{os.path.dirname(os.path.abspath(__file__))}/oracle_{name}.npz"
    np.savez(
        out,
        # exact model inputs (feed these verbatim to Core ML / MLX for parity):
        X=X.cpu().numpy(), S=S.cpu().numpy(), mask=mask.cpu().numpy(),
        chain_M=chain_M.cpu().numpy(), chain_M_pos=chain_M_pos.cpu().numpy(),
        residue_idx=residue_idx.cpu().numpy(),
        chain_encoding_all=chain_encoding_all.cpu().numpy(),
        E_idx=E_idx.cpu().numpy(),
        # reference outputs:
        log_p_uncond=log_p_uncond.cpu().numpy(),
        log_probs_fwd=log_probs_fwd.cpu().numpy(),
        native_score=native_score.cpu().numpy(),
        S_sample=S_sample.cpu().numpy(),
        seq=np.array(seq), native_seq=np.array(native_seq),
        k_neighbors=np.array(ckpt["num_edges"]),
    )
    print(f"[oracle] saved -> {out}")


if __name__ == "__main__":
    main()
