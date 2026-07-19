#!/usr/bin/env python3
"""
Repacking port - step 1: OpenFold side-chain GEOMETRY, reimplemented dependency-free.

Reimplements torsion_angles_to_frames + frames_and_literature_positions_to_atom14_pos
using ONLY explicit rotation-matrix / translation tensors -- no openfold, no Rigid
class. Crucially it replaces OpenFold's `frames[...,None,:] * one_hot(group) then sum`
(which broadcasts to rank 6 and is why Core ML rejected it) with a per-atom `gather`
of the correct group frame, keeping every tensor at rank <= 5.

This is the executable spec for the Swift/Metal (or MLX) port. `torsion_angles_to_frames_port`
and `frames_to_atom14_port` map 1:1 to Metal/MLX ops (matmul, gather, add, mul).

Constant tables are dumped to geometry_constants.npz for the native side.
"""
import os, sys
import numpy as np
import torch

LIG = "/Users/jcastellanos/repos/LigandMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIG)
from openfold.np.residue_constants import (  # noqa
    restype_rigid_group_default_frame, restype_atom14_to_rigid_group,
    restype_atom14_mask, restype_atom14_rigid_group_positions)


def make_backbone_frame_port(n_xyz, ca_xyz, c_xyz, eps=1e-9):
    """Backbone frame (R,t) per residue from N/CA/C, matching OpenFold's
    make_transform_from_reference (Gram-Schmidt via 3 axis rotations)."""
    trans = -ca_xyz
    n = n_xyz + trans
    c = c_xyz + trans
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]

    norm1 = torch.sqrt(eps + cx ** 2 + cy ** 2)
    sin_c1, cos_c1 = -cy / norm1, cx / norm1
    c1 = torch.zeros((*sin_c1.shape, 3, 3), dtype=n_xyz.dtype)
    c1[..., 0, 0] = cos_c1; c1[..., 0, 1] = -sin_c1
    c1[..., 1, 0] = sin_c1; c1[..., 1, 1] = cos_c1
    c1[..., 2, 2] = 1.0

    norm2 = torch.sqrt(eps + cx ** 2 + cy ** 2 + cz ** 2)
    sin_c2, cos_c2 = cz / norm2, torch.sqrt(cx ** 2 + cy ** 2) / norm2
    c2 = torch.zeros((*sin_c2.shape, 3, 3), dtype=n_xyz.dtype)
    c2[..., 0, 0] = cos_c2; c2[..., 0, 2] = sin_c2
    c2[..., 1, 1] = 1.0
    c2[..., 2, 0] = -sin_c2; c2[..., 2, 2] = cos_c2

    c_rots = c2 @ c1
    n = (c_rots @ n.unsqueeze(-1)).squeeze(-1)
    ny, nz = n[..., 1], n[..., 2]
    normn = torch.sqrt(eps + ny ** 2 + nz ** 2)
    sin_n, cos_n = -nz / normn, ny / normn
    n_rots = torch.zeros((*sin_n.shape, 3, 3), dtype=n_xyz.dtype)
    n_rots[..., 0, 0] = 1.0
    n_rots[..., 1, 1] = cos_n; n_rots[..., 1, 2] = -sin_n
    n_rots[..., 2, 1] = sin_n; n_rots[..., 2, 2] = cos_n

    rots = (n_rots @ c_rots).transpose(-1, -2)
    return rots, ca_xyz


def _compose(R1, t1, R2, t2):
    """Compose rigid transforms (R1,t1) ∘ (R2,t2). Batched over leading dims."""
    R = R1 @ R2
    t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


def torsion_angles_to_frames_port(bb_R, bb_t, alpha, aatype, rrgdf):
    """
    bb_R [B,N,3,3], bb_t [B,N,3] : backbone frame per residue.
    alpha [B,N,7,2]              : (sin,cos) of omega,phi,psi,chi1..chi4.
    aatype [B,N] long            : residue types (AF2 order).
    rrgdf [21,8,4,4]             : default group frames.
    returns global frames R_g [B,N,8,3,3], t_g [B,N,8,3].
    """
    B, N = aatype.shape
    d = rrgdf[aatype]                       # [B,N,8,4,4]
    dR, dt = d[..., :3, :3], d[..., :3, 3]  # [B,N,8,3,3], [B,N,8,3]

    # prepend backbone "rotation" [sin=0, cos=1] -> [B,N,8,2]
    bb_rot = alpha.new_zeros((B, N, 1, 2)); bb_rot[..., 1] = 1.0
    a = torch.cat([bb_rot, alpha], dim=-2)                       # [B,N,8,2]
    s, c = a[..., 0], a[..., 1]                                  # [B,N,8]

    # rotation about x-axis: [[1,0,0],[0,c,-s],[0,s,c]]
    R = torch.zeros(B, N, 8, 3, 3, dtype=alpha.dtype)
    R[..., 0, 0] = 1.0
    R[..., 1, 1] = c; R[..., 1, 2] = -s
    R[..., 2, 1] = s; R[..., 2, 2] = c

    # all_frames = default ∘ rot   (rot has zero translation)
    afR = dR @ R                                                 # [B,N,8,3,3]
    aft = dt                                                     # [B,N,8,3]

    # chain chi frames: 4 -> 5 -> 6 -> 7 (relative to backbone)
    chi1R, chi1t = afR[:, :, 4], aft[:, :, 4]
    chi2R, chi2t = _compose(chi1R, chi1t, afR[:, :, 5], aft[:, :, 5])
    chi3R, chi3t = _compose(chi2R, chi2t, afR[:, :, 6], aft[:, :, 6])
    chi4R, chi4t = _compose(chi3R, chi3t, afR[:, :, 7], aft[:, :, 7])

    bbR = afR.clone(); bbt = aft.clone()
    bbR[:, :, 5], bbt[:, :, 5] = chi2R, chi2t
    bbR[:, :, 6], bbt[:, :, 6] = chi3R, chi3t
    bbR[:, :, 7], bbt[:, :, 7] = chi4R, chi4t

    # to global: backbone ∘ frame  (broadcast backbone over the 8 groups)
    gR = bb_R.unsqueeze(2) @ bbR                                 # [B,N,8,3,3]
    gt = (bb_R.unsqueeze(2) @ bbt.unsqueeze(-1)).squeeze(-1) + bb_t.unsqueeze(2)
    return gR, gt


def frames_to_atom14_port(gR, gt, aatype, group_idx, atom_mask, lit_positions):
    """Place 14 atoms by gathering each atom's group frame (rank <= 5 throughout)."""
    grp = group_idx[aatype]                     # [B,N,14]  which of 8 groups
    # gather per-atom rotation + translation
    idxR = grp[..., None, None].expand(-1, -1, -1, 3, 3)        # [B,N,14,3,3]
    R_at = torch.gather(gR, 2, idxR)                            # [B,N,14,3,3]
    idxt = grp[..., None].expand(-1, -1, -1, 3)                 # [B,N,14,3]
    t_at = torch.gather(gt, 2, idxt)                            # [B,N,14,3]

    lit = lit_positions[aatype]                                 # [B,N,14,3]
    pos = (R_at @ lit.unsqueeze(-1)).squeeze(-1) + t_at         # [B,N,14,3]
    pos = pos * atom_mask[aatype].unsqueeze(-1)                 # [B,N,14,1]
    return pos


def dump_constants(path):
    np.savez(path,
             restype_rigid_group_default_frame=np.asarray(restype_rigid_group_default_frame, np.float32),
             restype_atom14_to_rigid_group=np.asarray(restype_atom14_to_rigid_group, np.int64),
             restype_atom14_mask=np.asarray(restype_atom14_mask, np.float32),
             restype_atom14_rigid_group_positions=np.asarray(restype_atom14_rigid_group_positions, np.float32))


if __name__ == "__main__":
    dump_constants(f"{HERE}/geometry_constants.npz")
    print(f"[geom] dumped constant tables -> {HERE}/geometry_constants.npz")
