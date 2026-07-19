#!/usr/bin/env python3
"""
End-to-end validation of the FULL MLX side-chain repack loop (mlx_repack_full.repack_full)
against a DETERMINISTIC variant of LigandMPNN's sc_utils.pack_side_chains.

The reference here is a copy of pack_side_chains / make_torsion_features with the two
sources of randomness removed so both sides are directly comparable:
  * chi initialisation: random -> deterministic 0  (chi angle 0 -> sin,cos = 0,1)
  * per-step selection : von-Mises-mixture SAMPLE -> MODE
        k = argmax(mix_logits, -1);  chi = mean[..., k]
Everything else (featurization, encode/decode weights, OpenFold geometry) is untouched.

For 6MRR and 5L33 (native sequence) we run BOTH pipelines for num_denoising_steps=3 and
report, over EXISTING side-chain atoms (atom14 indices 4..13 where xyz14_m==1):
  side-chain atom14 RMSD, max|diff|, %completeness (finite existing sc atoms), finite-check.

GATE: RMSD < 1e-2 A and max|diff| < 5e-2 A on both, 100% completeness, all finite.
"""
import os, sys
import numpy as np
import torch
import mlx.core as mx

LIG = "/Users/jcastellanos/repos/LigandMPNN"
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIG); sys.path.insert(0, HERE)

from data_utils import parse_PDB, featurize                         # noqa
from sc_utils import Packer, make_torsion_features, map_mpnn_to_af2_seq  # noqa
from openfold.utils import feats                                    # noqa
from openfold.utils.rigid_utils import Rigid                        # noqa
from openfold.data.data_transforms import atom37_to_torsion_angles, make_atom14_masks  # noqa
from openfold.np.residue_constants import (                         # noqa
    restype_rigid_group_default_frame, restype_atom14_to_rigid_group,
    restype_atom14_mask, restype_atom14_rigid_group_positions)
import torch.distributions as D                                     # noqa

import mlx_packer as M                                              # noqa
import mlx_repack_full as R                                         # noqa

DEVICE = torch.device("cpu")
SC_CKPT = f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt"
PDBS = {"6MRR": f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb",
        "5L33": f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb"}

RRGDF = torch.tensor(np.asarray(restype_rigid_group_default_frame), dtype=torch.float32)
GRP = torch.tensor(np.asarray(restype_atom14_to_rigid_group), dtype=torch.long)
AMASK = torch.tensor(np.asarray(restype_atom14_mask), dtype=torch.float32)
LIT = torch.tensor(np.asarray(restype_atom14_rigid_group_positions), dtype=torch.float32)


def build_model():
    ck = torch.load(SC_CKPT, map_location=DEVICE)
    m = Packer(node_features=128, edge_features=128, num_positional_embeddings=16,
               num_chain_embeddings=16, num_rbf=16, hidden_dim=128, num_encoder_layers=3,
               num_decoder_layers=3, atom_context_num=16, lower_bound=0.0, upper_bound=20.0,
               top_k=32, dropout=0.0, augment_eps=0.0, atom37_order=False, device=DEVICE, num_mix=3)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def make_torsion_features_det(feature_dict):
    """sc_utils.make_torsion_features (repack_everything=True) with chi init = 0."""
    device = feature_dict["mask"].device
    mask = feature_dict["mask"]
    B, L = mask.shape

    xyz37 = torch.zeros([B, L, 37, 3], device=device)
    xyz37[:, :, :3] = feature_dict["X"][:, :, :3]
    xyz37[:, :, 4] = feature_dict["X"][:, :, 3]

    S_af2 = torch.argmax(
        torch.nn.functional.one_hot(feature_dict["S"], 21).float()
        @ map_mpnn_to_af2_seq.to(device).float(), -1)
    masks14_37 = make_atom14_masks({"aatype": S_af2})
    torsion_dict = atom37_to_torsion_angles("")({
        "aatype": S_af2, "all_atom_positions": xyz37,
        "all_atom_mask": masks14_37["atom37_atom_exists"]})

    rigids = Rigid.make_transform_from_reference(
        n_xyz=xyz37[:, :, 0, :], ca_xyz=xyz37[:, :, 1, :], c_xyz=xyz37[:, :, 2, :], eps=1e-9)

    torsions_true = torch.zeros([B, L, 4, 2], device=device)
    mask_fix_sc = torch.ones([B, L, 1, 1], device=device)

    # DETERMINISTIC init: chi = 0 -> (sin,cos) = (0,1)
    init_sin_cos = torch.zeros([B, L, 4, 2], device=device)
    init_sin_cos[..., 1] = 1.0
    torsions_noised = torch.clone(torsion_dict["torsion_angles_sin_cos"])
    torsions_noised[:, :, 3:] = init_sin_cos * mask_fix_sc + torsions_true * (1 - mask_fix_sc)

    pred_frames = feats.torsion_angles_to_frames(rigids, torsions_noised, S_af2, RRGDF)
    xyz14_noised = feats.frames_and_literature_positions_to_atom14_pos(
        pred_frames, S_af2, RRGDF, GRP, AMASK, LIT)
    xyz14_m = masks14_37["atom14_atom_exists"] * mask[:, :, None]
    xyz14_noised = xyz14_noised * xyz14_m[:, :, :, None]

    torsion_dict.update(xyz14_m=xyz14_m, xyz14_noised=xyz14_noised, rigids=rigids,
                        torsions_noised=torsions_noised, mask_fix_sc=mask_fix_sc,
                        torsions_true=torsions_true, S_af2=S_af2)
    return torsion_dict


def pack_side_chains_det(feature_dict, model_sc, num_denoising_steps=3, num_context_atoms=16):
    """Deterministic (mode, chi=0 init, RNG-free) copy of sc_utils.pack_side_chains."""
    device = feature_dict["X"].device
    td = make_torsion_features_det(feature_dict)
    feature_dict["X"] = td["xyz14_noised"]
    feature_dict["X_m"] = td["xyz14_m"]
    for k in ("Y", "Y_t", "Y_m"):
        if k not in feature_dict:
            shp = [feature_dict["X"].shape[0], feature_dict["X"].shape[1], num_context_atoms]
            feature_dict[k] = torch.zeros(shp + ([3] if k == "Y" else []), device=device)
    h_V, h_E, E_idx = model_sc.encode(feature_dict)
    feature_dict["h_V"], feature_dict["h_E"], feature_dict["E_idx"] = h_V, h_E, E_idx

    for _ in range(num_denoising_steps):
        mean, concentration, mix_logits = model_sc.decode(feature_dict)
        k = torch.argmax(mix_logits, -1)                      # [B,L,4]
        sample = torch.gather(mean, -1, k[..., None])[..., 0]  # MODE angle [B,L,4]
        torsions_pred_unit = torch.cat(
            [torch.sin(sample[..., None]), torch.cos(sample[..., None])], -1)
        td["torsions_noised"][:, :, 3:] = (
            torsions_pred_unit * td["mask_fix_sc"] + td["torsions_true"] * (1 - td["mask_fix_sc"]))
        pred_frames = feats.torsion_angles_to_frames(
            td["rigids"], td["torsions_noised"], td["aatype"], RRGDF)
        xyz14_noised = feats.frames_and_literature_positions_to_atom14_pos(
            pred_frames, td["aatype"], RRGDF, GRP, AMASK, LIT)
        xyz14_noised = xyz14_noised * feature_dict["X_m"][:, :, :, None]
        feature_dict["X"] = xyz14_noised

    # b_factors from the mixture log-prob of the mode angles (mask_fix_sc == 1 everywhere)
    pred_dist = D.MixtureSameFamily(D.Categorical(logits=mix_logits),
                                    D.VonMises(mean, concentration))
    log_prob = pred_dist.log_prob(sample)                     # [B,L,4]
    tmp = GRP[td["S_af2"]].clone()
    tmp[tmp < 4] = 4; tmp -= 4
    b_factors = torch.gather(log_prob, -1, tmp)               # [B,L,14]
    return feature_dict["X"], feature_dict["X_m"], b_factors


def main():
    m = build_model()
    w = M.load_weights(SC_CKPT)
    print("Deterministic reference: sc_utils.pack_side_chains with sample->MODE, "
          "random chi init -> 0.  num_denoising_steps=3.\n")
    print(f"{'protein':8s}{'L':>5s}{'#sc_atoms':>10s} | {'RMSD(A)':>10s}{'max|diff|(A)':>13s}"
          f"{'complete%':>10s}{'finite':>7s}{'b_fac max|d|':>13s}  result")
    print("-" * 92)
    allok = True
    for name, pdb in PDBS.items():
        pd, *_ = parse_PDB(pdb, device=DEVICE, chains=[], parse_all_atoms=True)
        pd["chain_mask"] = torch.ones_like(pd["mask"])
        fd = featurize(pd, cutoff_for_score=8.0, use_atom_context=True,
                       number_of_ligand_atoms=16, model_type="ligand_mpnn")
        fd["S"] = fd["S"].long()

        X_bb = fd["X"][:, :, :4].detach().clone()
        S = fd["S"].detach().clone()
        mask = fd["mask"].detach().clone()
        R_idx = fd["R_idx"].detach().clone()
        chain_labels = fd["chain_labels"].detach().clone()

        # --- reference (deterministic) ---
        with torch.no_grad():
            ref_X, ref_Xm, ref_bf = pack_side_chains_det(dict(fd), m, num_denoising_steps=3)
        ref_X = ref_X.numpy(); ref_Xm = ref_Xm.numpy(); ref_bf = ref_bf.numpy()

        # --- MLX repack_full ---
        out = R.repack_full(w, X_bb.numpy(), S.numpy(), mask.numpy(),
                            R_idx.numpy(), chain_labels.numpy(), num_steps=3)
        mine = np.array(out["atom14"]); mine_m = np.array(out["atom14_mask"])
        mine_bf = np.array(out["b_factors"])

        # side-chain atoms = indices 4..13, existing where xyz14_m==1
        m_sc = ref_Xm[:, :, 4:]                                # [B,L,10]
        diff = (ref_X[:, :, 4:, :] - mine[:, :, 4:, :])
        sqd = (diff ** 2).sum(-1)                              # [B,L,10]
        nsc = float(m_sc.sum())
        rmsd = float(np.sqrt((m_sc * sqd).sum() / max(nsc, 1.0)))
        maxd = float(np.abs(diff * m_sc[..., None]).max())
        finite = bool(np.isfinite(mine).all())
        complete = 100.0 * float((np.isfinite(mine[:, :, 4:, :]).all(-1) * m_sc).sum()) / max(nsc, 1.0)
        bf_d = float(np.abs((ref_bf - mine_bf) * ref_Xm).max())

        ok = rmsd < 1e-2 and maxd < 5e-2 and complete >= 99.999 and finite
        allok &= ok
        print(f"{name:8s}{mask.shape[1]:5d}{int(nsc):10d} | {rmsd:10.3e}{maxd:13.3e}"
              f"{complete:10.2f}{str(finite):>7s}{bf_d:13.3e}  {'PASS' if ok else 'FAIL'}")
    print("-" * 92)
    print("[repack-full] GATE MET: side-chain atom14 RMSD<1e-2 & max|diff|<5e-2, 100% complete, finite"
          if allok else "[repack-full] GATE NOT MET")


if __name__ == "__main__":
    main()
