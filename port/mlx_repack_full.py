#!/usr/bin/env python3
"""
Repacking port - CAPSTONE: the FULL multi-step side-chain denoising loop in MLX.

Assembles the already-validated building blocks (mlx_features, mlx_packer,
mlx_geometry) into LigandMPNN's `sc_utils.pack_side_chains` end-to-end:

    backbone coords (N,CA,C,O) + MPNN sequence
      -> S_af2 remap + atom14 existence mask               (missing piece #2, here)
      -> backbone torsions omega/phi/psi via atom37_to_torsion_angles  (missing piece #1, here)
      -> backbone rigid frames + deterministic (chi=0) init atom14
      -> encode ONCE
      -> for step in range(num_steps):
             re-featurize decode from CURRENT atom14 X
             decode -> von-Mises mixture (mean, conc, mix_logits)
             MODE chi (argmax mixture component -> that component's mean)
             geometry: torsions -> frames -> atom14
      -> b_factors from the mixture log-prob of the mode angles

Deterministic (mode, chi=0 init, RNG-free). Ligand context is zeroed (protein-only).

Every op maps 1:1 to mlx-swift (matmul, gather/take_along_axis, erf, where, stack,
concatenate, sum/max, exp/log). This file is the executable spec for the Swift port.
"""
import os
import numpy as np
import mlx.core as mx

import mlx_packer as M
import mlx_features as MF
import mlx_geometry as MG

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- constant tables (shared with the geometry port) ----
_gc = np.load(f"{HERE}/geometry_constants.npz")
RRGDF = mx.array(_gc["restype_rigid_group_default_frame"].astype(np.float32))   # [21,8,4,4]
GRP = mx.array(_gc["restype_atom14_to_rigid_group"].astype(np.int32))           # [21,14]
AMASK = mx.array(_gc["restype_atom14_mask"].astype(np.float32))                 # [21,14]
LIT = mx.array(_gc["restype_atom14_rigid_group_positions"].astype(np.float32))  # [21,14,3]

# MPNN alphabet index -> OpenFold/AF2 restype index (from sc_utils.map_mpnn_to_af2_seq,
# a 21x21 permutation matrix; this is argmax over each row).
MPNN_TO_AF2 = mx.array(
    [0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20],
    dtype=mx.int32,
)

# periodic_table_features [3,119] from sc_utils.ProteinFeatures (row0=arange).
_PT_R1 = [0, 1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4,
          5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3,
          3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
_PT_R2 = [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6,
          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
          6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
          7, 7, 7, 7, 7, 7, 7]
PT = mx.array(np.stack([np.arange(119), np.array(_PT_R1), np.array(_PT_R2)]).astype(np.int32))

# von Mises log I0 polynomial coefficients (Abramowitz & Stegun; match torch.distributions).
_I0_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
             -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
LOG_2PI = float(np.log(2.0 * np.pi))


def _to_mx(x, dtype=None):
    a = x if isinstance(x, mx.array) else mx.array(np.asarray(x))
    return a.astype(dtype) if dtype is not None else a


def _cross(a, b):
    return mx.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _eval_poly(y, coef):
    r = coef[-1]
    for c in reversed(coef[:-1]):
        r = c + y * r
    return r


def log_i0(x):
    """log(I0(x)) for x > 0; matches torch.distributions VonMises._log_modified_bessel_fn."""
    y = (x / 3.75) ** 2
    small = mx.log(_eval_poly(y, _I0_SMALL))
    yl = 3.75 / x
    large = x - 0.5 * mx.log(x) + mx.log(_eval_poly(yl, _I0_LARGE))
    return mx.where(x < 3.75, small, large)


def _logsumexp(a, axis=-1):
    m = mx.max(a, axis=axis, keepdims=True)
    return (m + mx.log(mx.exp(a - m).sum(axis=axis, keepdims=True)))[..., 0] if axis == -1 \
        else m.squeeze(axis) + mx.log(mx.exp(a - m).sum(axis=axis))


def _dihedral_sin_cos(a0, a1, a2, a3, eps=1e-8):
    """One torsion via OpenFold's Rigid.from_3_points(p_neg_x=a1, origin=a2, p_xy=a0)
    then rel = R^T (a3 - a2); (sin,cos) = (rel_z, rel_y), normalized. Returns [...,2]."""
    e0 = a2 - a1
    e0 = e0 / mx.sqrt((e0 * e0).sum(-1, keepdims=True) + eps)
    e1 = a0 - a2
    e1 = e1 - e0 * (e0 * e1).sum(-1, keepdims=True)
    e1 = e1 / mx.sqrt((e1 * e1).sum(-1, keepdims=True) + eps)
    e2 = _cross(e0, e1)
    d = a3 - a2
    sin = (e2 * d).sum(-1)
    cos = (e1 * d).sum(-1)
    denom = mx.sqrt(sin * sin + cos * cos + 1e-8)
    return mx.stack([sin / denom, cos / denom], axis=-1)


def backbone_torsions(X_bb):
    """Extract (omega, phi, psi) sin/cos from backbone atoms N,CA,C,O.
    X_bb [B,L,4,3] (order N,CA,C,O). Returns [B,L,3,2]. Equivalent to OpenFold
    atom37_to_torsion_angles torsions 0..2 (with the psi sign flip)."""
    N, CA, C, O = X_bb[:, :, 0], X_bb[:, :, 1], X_bb[:, :, 2], X_bb[:, :, 3]
    z = mx.zeros_like(CA[:, :1])
    prevCA = mx.concatenate([z, CA[:, :-1]], axis=1)
    prevC = mx.concatenate([z, C[:, :-1]], axis=1)
    omega = _dihedral_sin_cos(prevCA, prevC, N, CA)     # a0,a1,a2,a3
    phi = _dihedral_sin_cos(prevC, N, CA, C)
    psi = _dihedral_sin_cos(N, CA, C, O)
    psi = psi * mx.array([-1.0, -1.0])                  # OpenFold sign vector [...,-1,...]
    return mx.stack([omega, phi, psi], axis=2)          # [B,L,3,2]


def _atom14_from_alpha(bb_R, bb_t, alpha, S_af2, xyz14_m):
    gR, gt = MG.torsion_angles_to_frames(bb_R, bb_t, alpha, S_af2, RRGDF)
    a14 = MG.frames_to_atom14(gR, gt, S_af2, GRP, AMASK, LIT)
    return a14 * xyz14_m[:, :, :, None]


def repack_full(w, X_bb, S, mask, R_idx, chain_labels, num_steps=3):
    """
    Full deterministic side-chain repack (MLX, mode / RNG-free, ligand context zeroed).

    Inputs (numpy or mlx arrays):
      w             : weights dict from mlx_packer.load_weights(sc_ckpt)
      X_bb  [1,L,4,3]: backbone coords, atom order N,CA,C,O
      S     [1,L]    : MPNN integer sequence (0..20)
      mask  [1,L]    : residue mask (1 present)
      R_idx [1,L]    : residue indices (for positional embeddings)
      chain_labels [1,L]: per-residue chain id
      num_steps      : denoising steps (default 3)

    Returns dict:
      atom14      [1,L,14,3] packed coordinates (backbone + side chains)
      atom14_mask [1,L,14]   existence mask (= AMASK[S_af2] * mask)  == xyz14_m
      b_factors   [1,L,14]   per-atom mixture log-prob (uncertainty)
    """
    X_bb = _to_mx(X_bb, mx.float32)
    S = _to_mx(S, mx.int32)
    mask = _to_mx(mask, mx.float32)
    R_idx = _to_mx(R_idx, mx.float32)
    chain_labels = _to_mx(chain_labels, mx.float32)
    B, L = S.shape

    # ---- missing piece #2: S_af2 remap + atom14 existence mask ----
    S_af2 = mx.take(MPNN_TO_AF2, S, axis=0)                      # [B,L] AF2 restype
    xyz14_m = mx.take(AMASK, S_af2, axis=0) * mask[:, :, None]   # [B,L,14]

    # ---- missing piece #1: backbone torsions omega/phi/psi ----
    bb_tors = backbone_torsions(X_bb)                            # [B,L,3,2]

    # backbone rigid frame (matches OpenFold make_transform_from_reference)
    bb_R, bb_t = MG.make_backbone_frame(X_bb[:, :, 0], X_bb[:, :, 1], X_bb[:, :, 2])

    # deterministic chi init = 0  -> (sin,cos)=(0,1)
    chi_init = mx.concatenate(
        [mx.zeros((B, L, 4, 1)), mx.ones((B, L, 4, 1))], axis=-1)  # [B,L,4,2]
    alpha0 = mx.concatenate([bb_tors, chi_init], axis=2)          # [B,L,7,2]
    atom14 = _atom14_from_alpha(bb_R, bb_t, alpha0, S_af2, xyz14_m)

    # zeroed ligand context (protein-only), atom_context_num = 16
    m_ctx = 16
    Y = mx.zeros((B, L, m_ctx, 3))
    Y_m = mx.zeros((B, L, m_ctx))
    Y_t = mx.zeros((B, L, m_ctx), dtype=mx.int32)

    # ---- encode ONCE (backbone-only featurization on the idealized atom14) ----
    V, E, E_idx, Y_nodes, Y_edges, E_context, Y_m_e = MF.features_encode(
        w, PT, S, atom14, Y, Y_m, Y_t, mask, R_idx, chain_labels)
    h_V, h_E = M.encode(w, V, E, E_idx, Y_nodes, Y_edges, E_context, Y_m_e, mask)

    mean = conc = mix = chi = None
    for _ in range(num_steps):
        dV, dF = MF.features_decode(w, S, atom14, xyz14_m, mask, E_idx, Y, Y_m, Y_t)
        mean, conc, mix = M.decode(w, dV, dF, h_V, h_E, E_idx, mask)   # [B,L,4,num_mix]
        # MODE: argmax mixture component -> that component's mean
        k = mx.argmax(mix, axis=-1)                                    # [B,L,4]
        chi = mx.take_along_axis(mean, k[..., None], axis=-1)[..., 0]  # [B,L,4]
        chi_sc = mx.stack([mx.sin(chi), mx.cos(chi)], axis=-1)         # [B,L,4,2]
        alpha = mx.concatenate([bb_tors, chi_sc], axis=2)             # [B,L,7,2]
        atom14 = _atom14_from_alpha(bb_R, bb_t, alpha, S_af2, xyz14_m)

    # ---- b_factors: mixture log-prob of the mode angles, mapped onto atom14 ----
    log_mix = mix - _logsumexp(mix, axis=-1)[..., None]               # log softmax
    vm = conc * mx.cos(chi[..., None] - mean) - LOG_2PI - log_i0(conc)  # [B,L,4,num_mix]
    log_prob = _logsumexp(log_mix + vm, axis=-1)                      # [B,L,4]
    grp = mx.take(GRP, S_af2, axis=0)                                 # [B,L,14]
    grp = mx.where(grp < 4, mx.array(4, mx.int32), grp) - 4           # backbone->chi1 slot
    b_factors = mx.take_along_axis(log_prob, grp, axis=-1)            # [B,L,14]

    return {"atom14": atom14, "atom14_mask": xyz14_m, "b_factors": b_factors}


if __name__ == "__main__":
    print("mlx_repack_full: import repack_full(w, X_bb, S, mask, R_idx, chain_labels, num_steps=3)")
