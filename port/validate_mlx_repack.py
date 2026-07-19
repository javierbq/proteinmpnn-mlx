#!/usr/bin/env python3
"""
Capstone: full assembled MLX repack pipeline (coords -> featurize -> encode -> decode
-> deterministic mode chi -> geometry -> atom14), compared to the same pipeline in
torch. Confirms all ported stages compose correctly end-to-end.

(Backbone torsions omega/phi/psi are taken from the captured structure; porting their
extraction, atom37_to_torsion_angles, is the one remaining mechanical dihedral calc.)
"""
import os, numpy as np, torch, mlx.core as mx
import mlx_packer as M, mlx_features as MF, mlx_geometry as MG
import geometry_port as GP
from validate_geometry_port import capture as geo_capture

HERE = os.path.dirname(os.path.abspath(__file__))
LIG = "/Users/jcastellanos/repos/LigandMPNN"
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
cap = np.load(f"{HERE}/packer_capture.npz")
w = M.load_weights(f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt")
pt = mx.array(cap["periodic_table_features"])
gc = np.load(f"{HERE}/geometry_constants.npz")
RRGDF = gc["restype_rigid_group_default_frame"]; GRP = gc["restype_atom14_to_rigid_group"]
AMASK = gc["restype_atom14_mask"]; LIT = gc["restype_atom14_rigid_group_positions"]

def A(n, d=None):
    return mx.array(cap[n].astype(d) if d else cap[n])

def mode_chi_np(mean, mix):
    k = np.argmax(mix, -1)                          # [B,N,4]
    chi = np.take_along_axis(mean, k[..., None], -1)[..., 0]
    return np.stack([np.sin(chi), np.cos(chi)], -1)  # [B,N,4,2]

def main():
    g = geo_capture(f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb")   # bb-torsions, aatype, NCAC (6MRR)
    bb_tors = g["alpha"].numpy()[:, :, :3, :]        # omega/phi/psi from structure
    aatype = g["aatype"].numpy().astype(np.int64)
    NCAC = g["NCAC"].numpy()

    # ---- MLX pipeline: featurize -> encode -> decode -> mode -> geometry ----
    V, E, E_idx, Yn, Ye, Ec, Ym = MF.features_encode(
        w, pt, A("enc_in_S", np.int32), A("enc_in_X"), A("enc_in_Y"), A("enc_in_Y_m"),
        A("enc_in_Y_t", np.int32), A("enc_in_mask"), A("enc_in_R_idx", np.float32),
        A("enc_in_chain_labels", np.float32))
    h_V, h_E = M.encode(w, V, E, E_idx, Yn, Ye, Ec, Ym, A("enc_in_mask"))
    dV, dF = MF.features_decode(w, A("dec_in_S", np.int32), A("dec_in_X"), A("dec_in_X_m"),
                               A("dec_in_mask"), A("dec_in_E_idx", np.int32), A("dec_in_Y"),
                               A("dec_in_Y_m"), A("dec_in_Y_t", np.int32))
    mean_m, _, mix_m = M.decode(w, dV, dF, h_V, h_E, E_idx, A("dec_in_mask"))
    chi_m = mode_chi_np(np.array(mean_m), np.array(mix_m))
    alpha_m = np.concatenate([bb_tors, chi_m], axis=2)
    pR, ptt = MG.make_backbone_frame(mx.array(NCAC[:, :, 0]), mx.array(NCAC[:, :, 1]), mx.array(NCAC[:, :, 2]))
    gR, gt = MG.torsion_angles_to_frames(pR, ptt, mx.array(alpha_m.astype(np.float32)),
                                         mx.array(aatype.astype(np.int32)), mx.array(RRGDF))
    atom14_mlx = np.array(MG.frames_to_atom14(gR, gt, mx.array(aatype.astype(np.int32)),
                          mx.array(GRP.astype(np.int32)), mx.array(AMASK), mx.array(LIT)))

    # ---- torch pipeline: captured decode -> mode -> torch geometry_port ----
    chi_t = mode_chi_np(cap["mean"], cap["mix_logits"])
    alpha_t = np.concatenate([bb_tors, chi_t], axis=2)
    tR, tt = GP.make_backbone_frame_port(torch.tensor(NCAC[:, :, 0]), torch.tensor(NCAC[:, :, 1]), torch.tensor(NCAC[:, :, 2]))
    gRt, gtt = GP.torsion_angles_to_frames_port(tR, tt, torch.tensor(alpha_t, dtype=torch.float32),
                                                torch.tensor(aatype), torch.tensor(RRGDF, dtype=torch.float32))
    atom14_t = GP.frames_to_atom14_port(gRt, gtt, torch.tensor(aatype), torch.tensor(GRP),
                                        torch.tensor(AMASK, dtype=torch.float32),
                                        torch.tensor(LIT, dtype=torch.float32)).numpy()

    md = float(np.abs(atom14_mlx - atom14_t).max())
    nan_free = bool(np.isfinite(atom14_mlx).all())
    print(f"[repack] full MLX pipeline vs torch pipeline: atom14 max|diff| = {md:.3e} A")
    print(f"[repack] MLX atom14 shape={atom14_mlx.shape}  finite={nan_free}")
    print("[repack] FULL MLX REPACK PIPELINE COMPOSES CORRECTLY" if md < 1e-4 and nan_free else "[repack] MISMATCH")

if __name__ == "__main__":
    main()
