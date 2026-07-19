#!/usr/bin/env python3
"""
Design port - PyTorch oracle capture for the LigandMPNN sequence-design model
(ligandmpnn_v_32_010_25.pt, model_type="ligand_mpnn", protein-only).

Uses LigandMPNN's OWN data pipeline (parse_PDB + featurize) so the feature_dict is
byte-for-byte what run.py produces, then dumps:
  - all featurization INPUTS (X, S_native, mask, R_idx, chain_labels, chain_mask,
    Y, Y_m, Y_t) and the fixed decoding order,
  - the canonical featurization OUTPUTS (V, E, E_idx, Y_nodes, Y_edges, Y_m),
  - the canonical encode OUTPUTS (h_V, h_E),
  - a deterministic GREEDY fixed-order decode (S, per-position logits, log_probs).

These are the parity targets for port/mlx_design.py (and, later, the Swift port).
Greedy fixed-order is RNG-free so it reproduces exactly across PyTorch / MLX / Swift.
"""
import os
import sys
import argparse
import numpy as np
import torch

LIGANDMPNN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LigandMPNN"))
sys.path.insert(0, LIGANDMPNN)

from model_utils import ProteinMPNN, cat_neighbors_nodes            # noqa: E402
from data_utils import parse_PDB, featurize                          # noqa: E402

CKPT = os.path.join(LIGANDMPNN, "model_params", "ligandmpnn_v_32_010_25.pt")


def build_feature_dict(pdb_path):
    protein_dict, _bb, _other, _icodes, _ = parse_PDB(pdb_path, device="cpu")
    L = protein_dict["R_idx"].shape[0]
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.float32)   # design everything
    fd = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=False,             # protein-only design (canonical ProteinMPNN; no ligand context)
        number_of_ligand_atoms=25,          # atom_context_num for this checkpoint
        model_type="ligand_mpnn",
    )
    fd["batch_size"] = 1
    fd["temperature"] = 0.1
    B, Lb, _, _ = fd["X"].shape
    fd["bias"] = torch.zeros(1, Lb, 21, dtype=torch.float32)
    fd["symmetry_residues"] = [[]]
    fd["symmetry_weights"] = [[]]
    return fd, Lb


def greedy_fixed_order_decode(model, fd, decoding_order):
    """Replica of ProteinMPNN.sample() (symmetry-free branch) but greedy (argmax,
    hard-omit X) with a FIXED decoding order. Returns S, logits[L,21], log_probs[L,21]."""
    S_true = fd["S"]
    mask = fd["mask"]
    chain_mask = fd["mask"] * fd["chain_mask"]
    device = S_true.device
    B, L = S_true.shape

    h_V, h_E, E_idx = model.encode(fd)

    # order-backward mask exactly as in sample()
    perm = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        perm, perm,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([B, L, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_S = torch.zeros_like(h_V)
    S = 20 * torch.ones((B, L), dtype=torch.int64, device=device)
    logits_out = torch.zeros((B, L, 21), dtype=torch.float32)
    logp_out = torch.zeros((B, L, 21), dtype=torch.float32)
    h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(model.decoder_layers))]

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    for t_ in range(L):
        t = decoding_order[:, t_]
        chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]
        mask_t = torch.gather(mask, 1, t[:, None])[:, 0]
        E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
        h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
        h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
        h_EXV_encoder_t = torch.gather(
            h_EXV_encoder_fw, 1,
            t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
        mask_bw_t = torch.gather(
            mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1]))
        for l, layer in enumerate(model.decoder_layers):
            h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
            h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
            h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
            h_V_stack[l + 1].scatter_(
                1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                layer(h_V_t, h_ESV_t, mask_V=mask_t))
        h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
        logits = model.W_out(h_V_t)                        # [B,21]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # greedy, hard-omit X (index 20): argmax over first 20 logits
        S_t = torch.argmax(logits[:, :20], dim=-1)
        S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
        S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
        logits_out[:, t[0]] = logits.float()
        logp_out[:, t[0]] = log_probs.float()
        h_S.scatter_(1, t[:, None, None].repeat(1, 1, h_S.shape[-1]), model.W_s(S_t)[:, None, :])
        S.scatter_(1, t[:, None], S_t[:, None])
    return S, logits_out, logp_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(CKPT, map_location="cpu")
    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, vocab=21,
        k_neighbors=ckpt["num_edges"], atom_context_num=ckpt["atom_context_num"],
        model_type="ligand_mpnn", device="cpu",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fd, L = build_feature_dict(args.pdb)

    # fixed, deterministic decoding order (RNG-free target)
    rng = np.random.RandomState(args.seed)
    randn = torch.tensor(np.abs(rng.randn(1, L)).astype(np.float32))
    chain_mask = fd["mask"] * fd["chain_mask"]
    decoding_order = torch.argsort((chain_mask + 0.0001) * randn)

    with torch.no_grad():
        V, E, E_idx, Y_nodes, Y_edges, Y_m = model.features(fd)
        h_V, h_E, E_idx2 = model.encode(fd)
        S, logits, log_probs = greedy_fixed_order_decode(model, fd, decoding_order)

    seq = "".join(["ACDEFGHIKLMNPQRSTVWYX"[i] for i in S[0].tolist()])
    print(f"[{args.id}] L={L}  greedy seq (first 60): {seq[:60]}")

    out = args.out or os.path.join(os.path.dirname(__file__), f"design_capture_{args.id}.npz")
    np.savez(
        out,
        # inputs
        X=fd["X"].numpy(), S_native=fd["S"].numpy(), mask=fd["mask"].numpy(),
        R_idx=fd["R_idx"].numpy(), chain_labels=fd["chain_labels"].numpy(),
        chain_mask=fd["chain_mask"].numpy(),
        Y=fd["Y"].numpy(), Y_m=fd["Y_m"].numpy(), Y_t=fd["Y_t"].numpy(),
        decoding_order=decoding_order.numpy(),
        # featurization outputs
        V=V.numpy(), E=E.numpy(), E_idx=E_idx.numpy(),
        Y_nodes=Y_nodes.numpy(), Y_edges=Y_edges.numpy(), Y_m_out=Y_m.numpy(),
        # encode outputs
        h_V=h_V.numpy(), h_E=h_E.numpy(),
        # decode outputs
        S=S.numpy(), logits=logits.numpy(), log_probs=log_probs.numpy(),
        seq=np.array(seq),
    )
    print(f"[{args.id}] wrote {out}")


if __name__ == "__main__":
    main()
