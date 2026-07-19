#!/usr/bin/env python3
"""
Repacking port - step 2: the Packer NN (encode + decode + torsion head) in MLX.

Reimplements the learned message-passing network in MLX (Apple's array framework;
same API as mlx-swift, so this is the executable spec for the Swift port). Takes
the featurization tensors as inputs (featurization port is step 3) and produces the
von-Mises-mixture torsion params (mean/concentration/mix_logits).

Ops used: matmul, add/mul, gelu, layer_norm, softplus, take_along_axis, concat, sum
-- all present in mlx-swift.
"""
import mlx.core as mx
import numpy as np
import torch

SQRT2 = 2.0 ** 0.5


def load_weights(pt_path):
    sd = torch.load(pt_path, map_location="cpu")["model_state_dict"]
    return {k: mx.array(v.float().numpy()) for k, v in sd.items()}


# ---- primitive ops (match PyTorch semantics) ----
def linear(w, name, x):
    W = w[f"{name}.weight"]                 # [out, in]
    y = x @ W.T
    b = w.get(f"{name}.bias")
    return y + b if b is not None else y

def layernorm(w, name, x, eps=1e-5):
    g, b = w[f"{name}.weight"], w[f"{name}.bias"]
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + eps) * g + b

def gelu(x):                                # exact erf form (torch.nn.GELU default)
    return 0.5 * x * (1.0 + mx.erf(x / SQRT2))

def softplus(x, beta=1.0, threshold=20.0):  # torch.nn.Softplus(beta,threshold)
    sx = beta * x
    return mx.where(sx > threshold, x, mx.log1p(mx.exp(sx)) / beta)


# ---- graph gather helpers ----
def gather_nodes(nodes, idx):               # nodes[B,N,C], idx[B,N,K] -> [B,N,K,C]
    B, N, C = nodes.shape
    K = idx.shape[-1]
    flat = mx.broadcast_to(idx.reshape(B, N * K, 1), (B, N * K, C))
    out = mx.take_along_axis(nodes, flat, axis=1)
    return out.reshape(B, N, K, C)

def gather_edges(edges, idx):               # edges[B,N,N,C], idx[B,N,K] -> [B,N,K,C]
    C = edges.shape[-1]
    nb = mx.broadcast_to(idx[..., None], (*idx.shape, C))
    return mx.take_along_axis(edges, nb, axis=2)

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    return mx.concatenate([h_neighbors, gather_nodes(h_nodes, E_idx)], axis=-1)


def ffn(w, name, h):                         # PositionWiseFeedForward
    return linear(w, f"{name}.W_out", gelu(linear(w, f"{name}.W_in", h)))


def enc_layer(w, p, h_V, h_E, E_idx, mask, mask_attend, scale=30.0):
    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_Vexp = mx.broadcast_to(h_V[..., None, :], h_EV.shape[:-1] + (h_V.shape[-1],))
    h_EV = mx.concatenate([h_Vexp, h_EV], axis=-1)
    hm = linear(w, f"{p}.W3", gelu(linear(w, f"{p}.W2", gelu(linear(w, f"{p}.W1", h_EV)))))
    hm = mask_attend[..., None] * hm
    dh = hm.sum(-2) / scale
    h_V = layernorm(w, f"{p}.norm1", h_V + dh)
    h_V = layernorm(w, f"{p}.norm2", h_V + ffn(w, f"{p}.dense", h_V))
    h_V = mask[..., None] * h_V
    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_Vexp = mx.broadcast_to(h_V[..., None, :], h_EV.shape[:-1] + (h_V.shape[-1],))
    h_EV = mx.concatenate([h_Vexp, h_EV], axis=-1)
    hm = linear(w, f"{p}.W13", gelu(linear(w, f"{p}.W12", gelu(linear(w, f"{p}.W11", h_EV)))))
    h_E = layernorm(w, f"{p}.norm3", h_E + hm)
    return h_V, h_E


def dec_layer(w, p, h_V, h_E, mask_V=None, mask_attend=None, scale=30.0):
    h_Vexp = mx.broadcast_to(h_V[..., None, :], h_E.shape[:-1] + (h_V.shape[-1],))
    h_EV = mx.concatenate([h_Vexp, h_E], axis=-1)
    hm = linear(w, f"{p}.W3", gelu(linear(w, f"{p}.W2", gelu(linear(w, f"{p}.W1", h_EV)))))
    if mask_attend is not None:
        hm = mask_attend[..., None] * hm
    dh = hm.sum(-2) / scale
    h_V = layernorm(w, f"{p}.norm1", h_V + dh)
    h_V = layernorm(w, f"{p}.norm2", h_V + ffn(w, f"{p}.dense", h_V))
    if mask_V is not None:
        h_V = mask_V[..., None] * h_V
    return h_V


def dec_layer_j(w, p, h_V, h_E, mask_V=None, mask_attend=None, scale=30.0):
    # like dec_layer but h_V expands over an extra (-2) axis to match h_E rank
    h_Vexp = mx.broadcast_to(h_V[..., None, :], h_E.shape[:-1] + (h_V.shape[-1],))
    h_EV = mx.concatenate([h_Vexp, h_E], axis=-1)
    hm = linear(w, f"{p}.W3", gelu(linear(w, f"{p}.W2", gelu(linear(w, f"{p}.W1", h_EV)))))
    if mask_attend is not None:
        hm = mask_attend[..., None] * hm
    dh = hm.sum(-2) / scale
    h_V = layernorm(w, f"{p}.norm1", h_V + dh)
    h_V = layernorm(w, f"{p}.norm2", h_V + ffn(w, f"{p}.dense", h_V))
    if mask_V is not None:
        h_V = mask_V[..., None] * h_V
    return h_V


def encode(w, V, E, E_idx, Y_nodes, Y_edges, E_context, Y_m, mask, n_enc=3, n_ctx=2):
    h_E_context = linear(w, "W_e_context", E_context)
    h_V = linear(w, "W_v", V)
    h_E = linear(w, "W_e", E)
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

    h_V_C = linear(w, "V_C", h_V_C)                 # no bias
    h_V = h_V + layernorm(w, "V_C_norm", h_V_C)
    return h_V, h_E


def decode(w, dec_V, dec_F, h_V, h_E, E_idx, mask, n_dec=3, num_mix=3):
    h_F = linear(w, "W_f", dec_F)
    h_EF = mx.concatenate([h_E, h_F], axis=-1)
    h_V_sc = linear(w, "W_v_sc", dec_V)
    h_V = linear(w, "linear_down", mx.concatenate([h_V, h_V_sc], axis=-1))
    for i in range(n_dec):
        h_EV = cat_neighbors_nodes(h_V, h_EF, E_idx)
        h_V = dec_layer(w, f"decoder_layers.{i}", h_V, h_EV, mask)
    t = linear(w, "W_torsions", h_V)
    B, N = h_V.shape[0], h_V.shape[1]
    t = t.reshape(B, N, 4, num_mix, 3)
    mean = t[:, :, :, :, 0]
    concentration = 0.1 + softplus(t[:, :, :, :, 1])
    mix_logits = t[:, :, :, :, 2]
    return mean, concentration, mix_logits
