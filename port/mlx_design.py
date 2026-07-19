#!/usr/bin/env python3
"""
Design port - LigandMPNN sequence-design model (ligandmpnn_v_32_010_25.pt,
model_type="ligand_mpnn", protein-only) reimplemented in MLX.

This is the executable spec for the mlx-swift port. It reuses the message-passing
primitives already validated in mlx_packer.py; only the featurization (different
weight keys + 2..22 A RBF) and the AUTOREGRESSIVE greedy decode are new here.

Pipeline:  features_design -> encode_design -> greedy_decode
  - featurization: ProteinFeaturesLigand.forward (use_side_chains=False)
  - encode:        ProteinMPNN.encode (ligand_mpnn branch; h_V starts at zeros)
  - greedy decode: ProteinMPNN.sample (symmetry-free branch) with argmax + fixed order

Protein-only simplification: no ligand atoms -> Y_m == 0 and Y_t == 0, so the
periodic-table group/period features are all zeros (element 0 == padding).
"""
import mlx.core as mx
import numpy as np
import torch

from mlx_packer import (linear, layernorm, gather_nodes, gather_edges,          # noqa
                        cat_neighbors_nodes, enc_layer, dec_layer, dec_layer_j)
from mlx_features import _cross, _normalize, _rbf, one_hot                        # noqa

CB = (-0.58273431, 0.56802827, -0.54067466)
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# design featurization RBF range (differs from the SC packer's 0..20)
RBF_LO, RBF_HI, NUM_RBF = 2.0, 22.0, 16

# exact ProteinFeaturesLigand RBF pair order (A query, B key); 25 pairs
_ATOM = {"N": 0, "Ca": 1, "C": 2, "O": 3, "Cb": 4}
_PAIRS = [
    ("Ca", "Ca"), ("N", "N"), ("C", "C"), ("O", "O"), ("Cb", "Cb"),
    ("Ca", "N"), ("Ca", "C"), ("Ca", "O"), ("Ca", "Cb"),
    ("N", "C"), ("N", "O"), ("N", "Cb"), ("Cb", "C"), ("Cb", "O"), ("O", "C"),
    ("N", "Ca"), ("C", "Ca"), ("O", "Ca"), ("Cb", "Ca"),
    ("C", "N"), ("O", "N"), ("Cb", "N"), ("C", "Cb"), ("O", "Cb"), ("C", "O"),
]


def load_design_weights(pt_path):
    sd = torch.load(pt_path, map_location="cpu")["model_state_dict"]
    return {k: mx.array(v.float().numpy()) for k, v in sd.items()}


def _get_rbf(A, B, E_idx):
    D = mx.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1) + 1e-6)   # [B,L,L]
    Dn = gather_edges(D[..., None], E_idx)[..., 0]                              # [B,L,K]
    return _rbf(Dn, RBF_LO, RBF_HI, NUM_RBF, [1, 1, 1, -1])


def _rbf_y(D):                                                                  # D [...]-> [...,NUM_RBF]
    return _rbf(D, RBF_LO, RBF_HI, NUM_RBF, [1] * D.ndim + [-1])


def dist_eidx(Ca, mask, top_k, eps=1e-6):
    m2 = mask[:, None, :] * mask[:, :, None]
    dX = Ca[:, None, :, :] - Ca[:, :, None, :]
    D = m2 * mx.sqrt((dX ** 2).sum(-1) + eps)
    Dm = D.max(-1, keepdims=True)
    Dadj = D + (1.0 - m2) * Dm
    k = min(top_k, Ca.shape[1])
    return mx.argsort(Dadj, axis=-1)[..., :k].astype(mx.int32)


def _make_angle_features(N, Ca, C, Y):
    v1, v2 = N - Ca, C - Ca
    e1 = _normalize(v1)
    u2 = v2 - e1 * (e1 * v2).sum(-1, keepdims=True)
    e2 = _normalize(u2)
    e3 = _cross(e1, e2)
    R = mx.stack([e1, e2, e3], axis=-1)              # [B,L,3,3] columns e1,e2,e3
    Vd = Y - Ca[:, :, None, :]                        # [B,L,M,3]
    loc = Vd @ R                                      # [B,L,M,3]
    rxy = mx.sqrt(loc[..., 0] ** 2 + loc[..., 1] ** 2 + 1e-8)
    rxyz = mx.sqrt((loc ** 2).sum(-1)) + 1e-8
    return mx.stack([loc[..., 0] / rxy, loc[..., 1] / rxy, rxy / rxyz, loc[..., 2] / rxyz], axis=-1)


def positional_embeddings(w, offset, echains, max_rel=32):
    d = mx.clip(offset + max_rel, 0, 2 * max_rel) * echains + (1 - echains) * (2 * max_rel + 1)
    return linear(w, "features.embeddings.linear", one_hot(d.astype(mx.int32), 2 * max_rel + 2))


def features_design(w, X, mask, R_idx, chain_labels, Y, Y_m, Y_t, top_k=32, e_idx=None):
    N, Ca, C, O = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
    b, c = Ca - N, C - Ca
    a = _cross(b, c)
    Cb = CB[0] * a + CB[1] * b + CB[2] * c + Ca
    atoms = {"N": N, "Ca": Ca, "C": C, "O": O, "Cb": Cb}
    E_idx = dist_eidx(Ca, mask, top_k) if e_idx is None else e_idx

    RBF_all = mx.concatenate([_get_rbf(atoms[a1], atoms[a2], E_idx) for a1, a2 in _PAIRS], axis=-1)

    offset = R_idx[:, :, None] - R_idx[:, None, :]
    offset = gather_edges(offset[..., None].astype(mx.float32), E_idx)[..., 0]
    dch = (chain_labels[:, :, None] - chain_labels[:, None, :] == 0).astype(mx.float32)
    Ech = gather_edges(dch[..., None], E_idx)[..., 0]
    E_pos = positional_embeddings(w, offset, Ech)
    E = layernorm(w, "features.norm_edges",
                  linear(w, "features.edge_embedding", mx.concatenate([E_pos, RBF_all], -1)))

    # ligand node/edge context (protein-only: Y_t == 0 -> group/period == 0)
    Y_t_1h = one_hot(Y_t, 120)
    Y_t_g_1h = one_hot(mx.zeros_like(Y_t), 19)
    Y_t_p_1h = one_hot(mx.zeros_like(Y_t), 8)
    Y1 = mx.concatenate([Y_t_1h, Y_t_g_1h, Y_t_p_1h], -1)             # [B,L,M,147]
    Y_t_lin = linear(w, "features.type_linear", Y1)                   # [B,L,M,64]

    rbfs = [_rbf_y(mx.sqrt(((at[:, :, None, :] - Y) ** 2).sum(-1) + 1e-6))
            for at in (N, Ca, C, O, Cb)]                             # 5 x [B,L,M,16]
    f_ang = _make_angle_features(N, Ca, C, Y)                        # [B,L,M,4]
    D_all = mx.concatenate(rbfs + [Y_t_lin, f_ang], -1)              # [B,L,M,148]
    V = layernorm(w, "features.norm_nodes", linear(w, "features.node_project_down", D_all))

    Yed = _rbf_y(mx.sqrt(((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2).sum(-1) + 1e-6))
    Y_edges = layernorm(w, "features.norm_y_edges", linear(w, "features.y_edges", Yed))
    Y_nodes = layernorm(w, "features.norm_y_nodes", linear(w, "features.y_nodes", Y1))
    return V, E, E_idx, Y_nodes, Y_edges, Y_m


def encode_design(w, V, E, E_idx, Y_nodes, Y_edges, Y_m, mask, n_enc=3, n_ctx=2):
    B, L = mask.shape
    h_V = mx.zeros((B, L, E.shape[-1]))
    h_E = linear(w, "W_e", E)
    h_E_context = linear(w, "W_v", V)

    mask_attend = gather_nodes(mask[..., None], E_idx)[..., 0]
    mask_attend = mask[..., None] * mask_attend
    for i in range(n_enc):
        h_V, h_E = enc_layer(w, f"encoder_layers.{i}", h_V, h_E, E_idx, mask, mask_attend)

    h_V_C = linear(w, "W_c", h_V)
    Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
    Y_nodes = linear(w, "W_nodes_y", Y_nodes)
    Y_edges = linear(w, "W_edges_y", Y_edges)
    for i in range(n_ctx):
        Y_nodes = dec_layer_j(w, f"y_context_encoder_layers.{i}", Y_nodes, Y_edges, Y_m, Y_m_edges)
        h_E_ctx_cat = mx.concatenate([h_E_context, Y_nodes], axis=-1)
        h_V_C = dec_layer(w, f"context_encoder_layers.{i}", h_V_C, h_E_ctx_cat, mask, Y_m)
    h_V_C = linear(w, "V_C", h_V_C)                                  # no bias
    h_V = h_V + layernorm(w, "V_C_norm", h_V_C)
    return h_V, h_E


def gather_nodes_g(nodes, idx):
    """General gather: nodes[B,Nn,C], idx[B,Nq,K] -> [B,Nq,K,C]. Unlike
    mlx_packer.gather_nodes (which assumes Nq==Nn), the query dim comes from idx,
    so this works for the autoregressive single-position case (Nq=1, Nn=L)."""
    B, Nn, C = nodes.shape
    Nq, K = idx.shape[1], idx.shape[2]
    flat = mx.broadcast_to(idx.reshape(B, Nq * K, 1), (B, Nq * K, C))
    out = mx.take_along_axis(nodes, flat, axis=1)
    return out.reshape(B, Nq, K, C)


def cat_neighbors_nodes_g(h_nodes, h_neighbors, E_idx):
    return mx.concatenate([h_neighbors, gather_nodes_g(h_nodes, E_idx)], axis=-1)


def _set_row(arr, t, val):
    """Replace column t (axis=1) of arr[B,L,...] with val[B,1,...] (host loop update)."""
    return mx.concatenate([arr[:, :t], val, arr[:, t + 1:]], axis=1)


def greedy_decode(w, h_V, h_E, E_idx, S_native, mask, chain_mask, decoding_order,
                  n_dec=3):
    B, L, C = h_V.shape
    order = [int(x) for x in np.asarray(decoding_order)[0].tolist()]

    perm = one_hot(mx.array(np.asarray(decoding_order)), L)          # [B,L,L]
    tri = 1.0 - mx.triu(mx.ones((L, L)))
    omb = mx.einsum("ij,biq,bjp->bqp", tri, perm, perm)             # [B,L,L]
    mask_attend = gather_edges(omb[..., None], E_idx)               # [B,L,K,1]
    mask_1D = mask.reshape(B, L, 1, 1)
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_S = mx.zeros((B, L, C))
    S = 20 * mx.ones((B, L), dtype=mx.int32)
    logits_out = mx.zeros((B, L, 21))
    logp_out = mx.zeros((B, L, 21))
    h_V_stack = [h_V] + [mx.zeros((B, L, C)) for _ in range(n_dec)]

    h_EX_encoder = cat_neighbors_nodes_g(mx.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes_g(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    Ws = w["W_s.weight"]                                             # [21,128]
    cm = (mask * chain_mask)                                         # [B,L]

    for t in order:
        E_idx_t = E_idx[:, t:t + 1]
        h_E_t = h_E[:, t:t + 1]
        h_ES_t = cat_neighbors_nodes_g(h_S, h_E_t, E_idx_t)
        h_EXV_encoder_t = h_EXV_encoder_fw[:, t:t + 1]
        mask_bw_t = mask_bw[:, t:t + 1]
        mask_t = mask[:, t:t + 1]                                    # [B,1]
        for l in range(n_dec):
            h_ESV_dec_t = cat_neighbors_nodes_g(h_V_stack[l], h_ES_t, E_idx_t)
            h_V_t = h_V_stack[l][:, t:t + 1]
            h_ESV_t = mask_bw_t * h_ESV_dec_t + h_EXV_encoder_t
            out = dec_layer(w, f"decoder_layers.{l}", h_V_t, h_ESV_t, mask_V=mask_t)
            h_V_stack[l + 1] = _set_row(h_V_stack[l + 1], t, out)

        h_V_t = h_V_stack[-1][:, t]                                  # [B,128]
        logits = linear(w, "W_out", h_V_t)                          # [B,21]
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        S_t = mx.argmax(logits[:, :20], axis=-1).astype(mx.int32)    # greedy, hard-omit X
        cm_t = cm[:, t].astype(mx.int32)
        S_true_t = S_native[:, t].astype(mx.int32)
        S_t = S_t * cm_t + S_true_t * (1 - cm_t)

        logits_out = _set_row(logits_out, t, logits[:, None, :])
        logp_out = _set_row(logp_out, t, log_probs[:, None, :])
        emb = mx.take(Ws, S_t, axis=0)[:, None, :]                   # [B,1,128]
        h_S = _set_row(h_S, t, emb)
        S = _set_row(S.reshape(B, L, 1), t, S_t.reshape(B, 1, 1)).reshape(B, L)
    return S, logits_out, logp_out


def design_greedy(w, X, S_native, mask, R_idx, chain_labels, Y, Y_m, Y_t,
                  chain_mask, decoding_order, top_k=32, e_idx=None):
    V, E, E_idx, Y_nodes, Y_edges, Y_m = features_design(
        w, X, mask, R_idx, chain_labels, Y, Y_m, Y_t, top_k=top_k, e_idx=e_idx)
    h_V, h_E = encode_design(w, V, E, E_idx, Y_nodes, Y_edges, Y_m, mask)
    S, logits, log_probs = greedy_decode(w, h_V, h_E, E_idx, S_native, mask, chain_mask, decoding_order)
    return dict(V=V, E=E, E_idx=E_idx, Y_nodes=Y_nodes, Y_edges=Y_edges,
                h_V=h_V, h_E=h_E, S=S, logits=logits, log_probs=log_probs)
