#!/usr/bin/env python3
"""
Repacking port - step 3: the Packer FEATURIZATION (features_encode / features_decode) in MLX.

Reproduces the ligand-aware featurization: kNN graph, backbone RBFs, positional/chain
embeddings, ligand-atom context (periodic-table one-hots, distance RBFs, local-frame
angle features), and the decode-time side-chain distance features. All ops are
mlx-swift-available. Weight prefix "features.".
"""
import mlx.core as mx
from mlx_packer import linear, layernorm, gather_nodes, gather_edges  # noqa

CB = (-0.58273431, 0.56802827, -0.54067466)


def one_hot(idx, n):
    return (idx[..., None] == mx.arange(n)).astype(mx.float32)


def _cross(a, b):
    return mx.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _normalize(v, eps=1e-12):
    n = mx.sqrt((v * v).sum(-1, keepdims=True))
    return v / mx.maximum(n, eps)


def _rbf(D, lo, hi, num_bins, mu_shape):
    mu = mx.linspace(lo, hi, num_bins).reshape(mu_shape)
    sigma = (hi - lo) / num_bins
    return mx.exp(-(((D[..., None] - mu) / sigma) ** 2))


def _get_rbf(A, B, E_idx, lo, hi, num_bins):
    D = mx.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1) + 1e-6)   # [B,L,L]
    Dn = gather_edges(D[..., None], E_idx)[..., 0]                             # [B,L,K]
    return _rbf(Dn, lo, hi, num_bins, [1, 1, 1, -1])


def dist_eidx(Ca, mask, top_k, eps=1e-6):
    m2 = mask[:, None, :] * mask[:, :, None]
    dX = Ca[:, None, :, :] - Ca[:, :, None, :]
    D = m2 * mx.sqrt((dX ** 2).sum(-1) + eps)
    Dm = D.max(-1, keepdims=True)
    Dadj = D + (1.0 - m2) * Dm
    k = min(top_k, Ca.shape[1])
    return mx.argsort(Dadj, axis=-1)[..., :k].astype(mx.int32)


def positional_embeddings(w, offset, echains, max_rel=32):
    d = mx.clip(offset + max_rel, 0, 2 * max_rel) * echains + (1 - echains) * (2 * max_rel + 1)
    return linear(w, "features.positional_embeddings.linear", one_hot(d, 2 * max_rel + 2))


def _make_angle_features(N, Ca, C, Y):
    v1, v2 = N - Ca, C - Ca
    e1 = _normalize(v1)
    u2 = v2 - e1 * (e1 * v2).sum(-1, keepdims=True)
    e2 = _normalize(u2)
    e3 = _cross(e1, e2)
    R = mx.stack([e1, e2, e3], axis=-1)          # [B,L,3,3], last axis = basis
    Vd = Y - Ca[:, :, None, :]                    # [B,L,M,3]
    loc = Vd @ R                                  # [B,L,M,3]
    rxy = mx.sqrt(loc[..., 0] ** 2 + loc[..., 1] ** 2 + 1e-8)
    rxyz = mx.sqrt((loc ** 2).sum(-1)) + 1e-8
    return mx.stack([loc[..., 0] / rxy, loc[..., 1] / rxy, rxy / rxyz, loc[..., 2] / rxyz], axis=-1)


def features_encode(w, pt, S, X, Y, Y_m, Y_t, mask, R_idx, chain_labels,
                    top_k=32, num_rbf=16, lo=0.0, hi=20.0, e_idx=None):
    N, Ca, C, O = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
    b, c = Ca - N, C - Ca
    a = _cross(b, c)
    Cb = CB[0] * a + CB[1] * b + CB[2] * c + Ca
    E_idx = dist_eidx(Ca, mask, top_k) if e_idx is None else e_idx

    atoms = [N, Ca, C, O, Cb]
    RBF_all = mx.concatenate([_get_rbf(a1, a2, E_idx, lo, hi, num_rbf)
                              for a1 in atoms for a2 in atoms], axis=-1)

    offset = R_idx[:, :, None] - R_idx[:, None, :]
    offset = gather_edges(offset[..., None].astype(mx.float32), E_idx)[..., 0]
    dch = (chain_labels[:, :, None] - chain_labels[:, None, :] == 0).astype(mx.float32)
    Ech = gather_edges(dch[..., None], E_idx)[..., 0]
    E_pos = positional_embeddings(w, offset, Ech)
    E = layernorm(w, "features.enc_norm_edges", linear(w, "features.enc_edge_embedding",
                  mx.concatenate([E_pos, RBF_all], -1)))
    V = layernorm(w, "features.enc_norm_nodes", linear(w, "features.enc_node_embedding",
                  one_hot(S, 21)))

    # ligand context
    Y_t_g = mx.take(pt[1], Y_t, axis=0)
    Y_t_p = mx.take(pt[2], Y_t, axis=0)
    Y1 = mx.concatenate([one_hot(Y_t, 120), one_hot(Y_t_g, 19), one_hot(Y_t_p, 8)], -1)  # [B,L,M,147]
    Y_t_lin = linear(w, "features.type_linear", Y1)
    rbfs = [_rbf(mx.sqrt(((at[:, :, None, :] - Y) ** 2).sum(-1) + 1e-6), lo, hi, num_rbf, [1, 1, 1, -1])
            for at in (N, Ca, C, O, Cb)]
    f_ang = _make_angle_features(N, Ca, C, Y)
    D_all = mx.concatenate(rbfs + [Y_t_lin, f_ang], -1)
    E_context = layernorm(w, "features.norm_nodes", linear(w, "features.node_project_down", D_all))

    Yed = _rbf(mx.sqrt(((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2).sum(-1) + 1e-6),
               0.0, 20.0, num_rbf, [1, 1, 1, 1, -1])
    Y_edges = layernorm(w, "features.norm_y_edges", linear(w, "features.y_edges", Yed))
    Y_nodes = layernorm(w, "features.norm_y_nodes", linear(w, "features.y_nodes", Y1))
    return V, E, E_idx, Y_nodes, Y_edges, E_context, Y_m


def features_decode(w, S, X, X_m, mask, E_idx, Y, Y_m, Y_t,
                    atom_context_num=16, num_rbf=16, lo=0.0, hi=20.0):
    Y = Y[:, :, :atom_context_num]; Y_m = Y_m[:, :, :atom_context_num]; Y_t = Y_t[:, :, :atom_context_num]
    X_m = X_m * mask[:, :, None]
    B, L = X.shape[0], X.shape[1]
    X_m_g = gather_nodes(X_m, E_idx)                                  # [B,L,K,14]

    RBF_sc = []
    for i in range(14):
        for j in range(14):
            r = _get_rbf(X[:, :, i, :], X[:, :, j, :], E_idx, lo, hi, num_rbf)
            r = r * X_m[:, :, i, None, None] * X_m_g[:, :, :, j, None]
            RBF_sc.append(r)

    D_XY = mx.sqrt(((X[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2).sum(-1) + 1e-6)   # [B,L,14,M]
    XY = _rbf(D_XY, lo, hi, num_rbf, [1, 1, 1, 1, -1])
    XY = XY * X_m[:, :, :, None, None] * Y_m[:, :, None, :, None]
    Y_t_1h = one_hot(Y_t, 120)                                        # [B,L,M,120]
    Y_t_1h = mx.broadcast_to(Y_t_1h[:, :, None], (B, L, 14, Y.shape[2], 120))
    XY = mx.concatenate([XY, Y_t_1h], -1)
    XY = linear(w, "features.W_XY_project_down1", XY).reshape(B, L, -1)
    V = layernorm(w, "features.dec_norm_nodes1", linear(w, "features.dec_node_embedding1", XY))

    S1 = one_hot(S, 21)
    S1g = gather_nodes(S1, E_idx)                                     # [B,L,K,21]
    S1e = mx.broadcast_to(S1[:, :, None, :], (B, L, E_idx.shape[2], 21))
    S_feat = mx.concatenate([S1e, S1g], -1)                          # [B,L,K,42]
    F = mx.concatenate(RBF_sc + [S_feat], -1)
    F = layernorm(w, "features.dec_norm_edges1", linear(w, "features.dec_edge_embedding1", F))
    return V, F
