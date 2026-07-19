#!/usr/bin/env python3
"""
Repacking port - geometry in MLX (transcribed from geometry_port.py, rank <= 5).
Backbone frame from N/CA/C, torsion angles -> global rigid frames, frames -> atom14.
Uses mx.stack (not in-place assignment) so it's clean/export-friendly.
"""
import mlx.core as mx


def _R_x(c, s):
    """rotation about x-axis [[1,0,0],[0,c,-s],[0,s,c]], batched; c,s shape [...]."""
    z = mx.zeros_like(c); o = mx.ones_like(c)
    r0 = mx.stack([o, z, z], axis=-1)
    r1 = mx.stack([z, c, -s], axis=-1)
    r2 = mx.stack([z, s, c], axis=-1)
    return mx.stack([r0, r1, r2], axis=-2)


def _compose(R1, t1, R2, t2):
    return R1 @ R2, (R1 @ t2[..., None])[..., 0] + t1


def make_backbone_frame(N, Ca, C, eps=1e-9):
    trans = -Ca
    n = N + trans; c = C + trans
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    z = mx.zeros_like(cx); o = mx.ones_like(cx)

    n1 = mx.sqrt(eps + cx ** 2 + cy ** 2)
    s1, c1 = -cy / n1, cx / n1
    C1 = mx.stack([mx.stack([c1, -s1, z], -1), mx.stack([s1, c1, z], -1), mx.stack([z, z, o], -1)], -2)

    n2 = mx.sqrt(eps + cx ** 2 + cy ** 2 + cz ** 2)
    s2, c2 = cz / n2, mx.sqrt(cx ** 2 + cy ** 2) / n2
    C2 = mx.stack([mx.stack([c2, z, s2], -1), mx.stack([z, o, z], -1), mx.stack([-s2, z, c2], -1)], -2)

    c_rots = C2 @ C1
    n = (c_rots @ n[..., None])[..., 0]
    ny, nz = n[..., 1], n[..., 2]
    nn = mx.sqrt(eps + ny ** 2 + nz ** 2)
    sn, cn = -nz / nn, ny / nn
    Nr = mx.stack([mx.stack([o, z, z], -1), mx.stack([z, cn, -sn], -1), mx.stack([z, sn, cn], -1)], -2)

    rots = (Nr @ c_rots).swapaxes(-1, -2)
    return rots, Ca


def torsion_angles_to_frames(bb_R, bb_t, alpha, aatype, rrgdf):
    B, Nres = aatype.shape
    d = rrgdf[aatype]                       # [B,N,8,4,4]
    dR, dt = d[..., :3, :3], d[..., :3, 3]

    bb_rot = mx.concatenate([mx.zeros((B, Nres, 1, 1)), mx.ones((B, Nres, 1, 1))], axis=-1)
    a = mx.concatenate([bb_rot, alpha], axis=-2)        # [B,N,8,2]
    R = _R_x(a[..., 1], a[..., 0])                       # (c=a1, s=a0)

    afR = dR @ R
    aft = dt
    c1R, c1t = afR[:, :, 4], aft[:, :, 4]
    c2R, c2t = _compose(c1R, c1t, afR[:, :, 5], aft[:, :, 5])
    c3R, c3t = _compose(c2R, c2t, afR[:, :, 6], aft[:, :, 6])
    c4R, c4t = _compose(c3R, c3t, afR[:, :, 7], aft[:, :, 7])

    # replace groups 5,6,7 with chained chi frames (build via concatenate, no in-place)
    bbR = mx.concatenate([afR[:, :, :5], c2R[:, :, None], c3R[:, :, None], c4R[:, :, None]], axis=2)
    bbt = mx.concatenate([aft[:, :, :5], c2t[:, :, None], c3t[:, :, None], c4t[:, :, None]], axis=2)

    gR = bb_R[:, :, None] @ bbR
    gt = (bb_R[:, :, None] @ bbt[..., None])[..., 0] + bb_t[:, :, None]
    return gR, gt


def frames_to_atom14(gR, gt, aatype, group_idx, atom_mask, lit):
    grp = group_idx[aatype]                              # [B,N,14]
    idxR = mx.broadcast_to(grp[..., None, None], (*grp.shape, 3, 3))
    R_at = mx.take_along_axis(gR, idxR, axis=2)
    idxt = mx.broadcast_to(grp[..., None], (*grp.shape, 3))
    t_at = mx.take_along_axis(gt, idxt, axis=2)
    litp = lit[aatype]                                   # [B,N,14,3]
    pos = (R_at @ litp[..., None])[..., 0] + t_at
    return pos * atom_mask[aatype][..., None]
