#!/usr/bin/env python3
"""Capture reference per-position log-probs (3 ScoreMode semantics) to .safetensors
oracle fixtures for the Swift MPNNKit score() parity tests. Mirrors capture_design.py.

The scoring model is LigandMPNN's model_utils.ProteinMPNN loaded from
ligandmpnn_v_32_010_25.pt (model_type="ligand_mpnn", use_atom_context=False,
protein-only) -- the SAME loaded model + feature_dict as capture_design.py, so the
weights match the Swift side.

IMPORTANT (verified 2026-07-20): that LigandMPNN class exposes only
`score(feature_dict, use_sequence)` and `single_aa_score(feature_dict, use_sequence)`
(model_utils.py:560, :471). It does NOT expose the cleanly-named
`forward`/`unconditional_probs`/`conditional_probs` -- those live on the *separate*
ProteinMPNN architecture in ../ProteinMPNN/protein_mpnn_utils.py. The encode half
differs between the two classes (LigandMPNN has a ligand-context encoder, inert with
use_atom_context=False), but the *decode* half is byte-identical (same DecLayer,
cat_neighbors_nodes, W_s, W_out and mask_bw/mask_fw machinery). So we reuse the loaded
LigandMPNN `model.encode(fd)` (weights == Swift) and reimplement the three decode passes
here, mirroring the canonical ProteinMPNN reference exactly:

  conditional     <- ProteinMPNN.forward(use_input_decoding_order=True, decoding_order)
                     (protein_mpnn_utils.py:1057-1100). == LigandMPNN score(use_sequence=True).
  unconditional   <- ProteinMPNN.unconditional_probs (protein_mpnn_utils.py:1352-1382).
                     STRICT zeroed order_mask_backward -> order-independent, structure only.
                     NOT LigandMPNN score(use_sequence=False), which still gates the forward
                     encoder embeddings by a random decode order.
  leaveOneOut     <- ProteinMPNN.conditional_probs(backbone_only=False)
                     (protein_mpnn_utils.py:1292-1349). Per idx: order_mask=zeros; order_mask[idx]=1
                     places idx LAST in the decode order -> it sees ALL other native residues.
                     Correctly named, so it sidesteps the single_aa_score use_sequence
                     flag-inversion trap (model_utils.py:502-507).

Reimplementing the decode locally (rather than calling model.score/single_aa_score) lets us
(a) pass an explicit fixed decoding_order for `conditional` and record it, and (b) get the
strict-zeroed `unconditional` semantics that no LigandMPNN method exposes.

Alphabet column order is ACDEFGHIKLMNPQRSTVWYX (index 20 == X).
"""
import os
import sys
import argparse
import numpy as np
import torch
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
LIGANDMPNN = os.path.abspath(os.path.join(HERE, "..", "..", "LigandMPNN"))
sys.path.insert(0, LIGANDMPNN)
from model_utils import ProteinMPNN, cat_neighbors_nodes              # noqa: E402
sys.path.insert(0, HERE)
from capture_design import build_feature_dict, CKPT                   # noqa: E402

ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "MPNNBench", "Resources", "app_assets"))
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def load_model():
    """Exact load used by capture_design.py: LigandMPNN ProteinMPNN, ligand_mpnn,
    protein-only (use_atom_context off via featurize), eval mode."""
    ckpt = torch.load(CKPT, map_location="cpu")
    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, vocab=21,
        k_neighbors=ckpt["num_edges"], atom_context_num=ckpt["atom_context_num"],
        model_type="ligand_mpnn", device="cpu",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _order_mask_backward(decoding_order, L, device):
    """order_mask_backward[b,q,p] = 1 iff position q is decoded AFTER position p."""
    perm = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
    return torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        perm, perm,
    )


def score_conditional(model, fd, decoding_order):
    """Teacher-forced autoregressive log-probs over ONE fixed decode order, native S
    fed. Each position sees residues earlier in the order. Deterministic.
    Mirrors ProteinMPNN.forward(use_input_decoding_order=True) (protein_mpnn_utils.py:1057)."""
    S = fd["S"]
    mask = fd["mask"]
    B, L = S.shape
    device = S.device

    h_V, h_E, E_idx = model.encode(fd)
    h_S = model.W_s(S)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    order_mask_backward = _order_mask_backward(decoding_order, L, device)
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([B, L, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    for layer in model.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)

    logits = model.W_out(h_V)
    return torch.nn.functional.log_softmax(logits, dim=-1)


def score_unconditional(model, fd):
    """Structure-only log-probs, no sequence conditioning. STRICT zeroed backward-order
    mask -> every position attends to ALL neighbor backbone embeddings, order-independent.
    Mirrors ProteinMPNN.unconditional_probs (protein_mpnn_utils.py:1352)."""
    S = fd["S"]
    mask = fd["mask"]
    B, L = S.shape
    device = S.device

    h_V, h_E, E_idx = model.encode(fd)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    order_mask_backward = torch.zeros([B, L, L], device=device)  # strict zero
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([B, L, 1, 1])
    mask_fw = mask_1D * (1.0 - mask_attend)  # == mask_1D

    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    for layer in model.decoder_layers:
        h_V = layer(h_V, h_EXV_encoder_fw, mask)

    logits = model.W_out(h_V)
    return torch.nn.functional.log_softmax(logits, dim=-1)


def score_leaveoneout(model, fd, randn):
    """Each position conditioned on ALL other native residues. Per idx, order_mask is
    zero everywhere except idx (=1), so argsort places idx LAST in the decode order ->
    idx sees every other position (all backward). Order-independent by construction.
    Mirrors ProteinMPNN.conditional_probs(backbone_only=False) (protein_mpnn_utils.py:1292).
    (backbone_only=False == leave-one-out; the LigandMPNN single_aa_score(use_sequence)
    flag is inverted vs its help text, model_utils.py:502-507 -- this correctly-named path
    avoids that trap.)"""
    S = fd["S"]
    mask = fd["mask"]
    chain_mask = fd["mask"] * fd["chain_mask"]
    B, L = S.shape
    device = S.device

    h_V_enc, h_E, E_idx = model.encode(fd)
    h_S = model.W_s(S)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

    idx_to_loop = np.argwhere(chain_mask.cpu().numpy()[0, :] == 1)[:, 0]
    out = torch.zeros([B, L, 21], device=device).float()
    mask_1D = mask.view([B, L, 1, 1])

    for idx in idx_to_loop:
        h_V = torch.clone(h_V_enc)
        order_mask = torch.zeros(L, device=device).float()
        order_mask[idx] = 1.0  # idx decoded LAST -> conditioned on all other residues
        decoding_order = torch.argsort((order_mask[None,] + 0.0001) * torch.abs(randn))

        order_mask_backward = _order_mask_backward(decoding_order, L, device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in model.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = model.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        out[:, idx, :] = log_probs[:, idx, :]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_model()
    fd, L = build_feature_dict(args.pdb)
    S = fd["S"]  # [1,L] i64 native sequence

    # fixed, deterministic decode order for `conditional` (chain_mask all ones -> the
    # (chain_mask+1e-4) factor only scales, argsort unchanged; matches export_app_assets
    # decoding_order_for(seed) and capture_design.py's order convention).
    rng = np.random.RandomState(args.seed)
    randn = torch.tensor(np.abs(rng.randn(1, L)).astype(np.float32))
    chain_mask = fd["mask"] * fd["chain_mask"]
    decoding_order = torch.argsort((chain_mask + 0.0001) * randn)  # [1,L]

    with torch.no_grad():
        cond = score_conditional(model, fd, decoding_order)
        uncond = score_unconditional(model, fd)
        loo = score_leaveoneout(model, fd, randn)

    out = {
        "native_seq":             mx.array(S[0].cpu().numpy().astype(np.int32)),
        "logprobs_conditional":   mx.array(cond[0].cpu().numpy().astype(np.float32)),
        "logprobs_unconditional": mx.array(uncond[0].cpu().numpy().astype(np.float32)),
        "logprobs_leaveoneout":   mx.array(loo[0].cpu().numpy().astype(np.float32)),
        "decoding_order":         mx.array(decoding_order[0].cpu().numpy().astype(np.int32)),
    }
    os.makedirs(f"{ASSETS}/oracles", exist_ok=True)
    path = f"{ASSETS}/oracles/{args.id}_score.safetensors"
    mx.save_safetensors(path, out, metadata={"format": "mpnnbench"})
    seq = "".join(ALPHABET[i] for i in S[0].tolist())
    print(f"[capture_score] {args.id} L={L} -> oracles/{args.id}_score.safetensors  seq[:40]={seq[:40]}")


if __name__ == "__main__":
    main()
