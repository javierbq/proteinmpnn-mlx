#!/usr/bin/env python3
"""Capture all PyTorch Packer intermediates as fixtures for validating the MLX port.
Saves featurization outputs (encode+decode) and layer outputs so each ported stage
can be checked independently."""
import os, sys
import numpy as np
import torch

LIG = "/Users/jcastellanos/repos/LigandMPNN"
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIG)
from data_utils import parse_PDB, featurize          # noqa
from sc_utils import Packer, pack_side_chains          # noqa

DEVICE = torch.device("cpu")
SC_CKPT = f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt"
PDB = f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb"


def main():
    pd, *_ = parse_PDB(PDB, device=DEVICE, chains=[], parse_all_atoms=True)
    pd["chain_mask"] = torch.ones_like(pd["mask"])
    ck = torch.load(SC_CKPT, map_location=DEVICE)
    m = Packer(node_features=128, edge_features=128, num_positional_embeddings=16,
               num_chain_embeddings=16, num_rbf=16, hidden_dim=128, num_encoder_layers=3,
               num_decoder_layers=3, atom_context_num=16, lower_bound=0.0, upper_bound=20.0,
               top_k=32, dropout=0.0, augment_eps=0.0, atom37_order=False, device=DEVICE, num_mix=3)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    fd = featurize(pd, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=16, model_type="ligand_mpnn")
    fd["S"] = fd["S"].long()

    cap = {}
    fe, fdle = m.features.features_encode, m.features.features_decode
    enc, dec = m.encode, m.decode

    def fe_hook(feature_dict):
        # raw inputs to features_encode
        for k in ("S", "X", "Y", "Y_m", "Y_t", "mask", "R_idx", "chain_labels"):
            cap[f"enc_in_{k}"] = feature_dict[k]
        V, E, E_idx, Yn, Ye, Ec, Ym = fe(feature_dict)
        cap.update(dict(V=V, E=E, E_idx=E_idx, Y_nodes=Yn, Y_edges=Ye, E_context=Ec, Y_m=Ym,
                        mask=feature_dict["mask"]))
        return V, E, E_idx, Yn, Ye, Ec, Ym

    def enc_hook(feature_dict):
        hV, hE, Ei = enc(feature_dict)
        cap.update(dict(h_V=hV, h_E=hE))
        return hV, hE, Ei

    def fd_hook(feature_dict):
        if "dec_in_X" not in cap:  # capture first denoising step's decode inputs
            for k in ("S", "X", "X_m", "mask", "E_idx", "Y", "Y_m", "Y_t"):
                cap[f"dec_in_{k}"] = feature_dict[k].detach().clone()
        V, F = fdle(feature_dict)
        if "dec_V" not in cap:
            cap.update(dict(dec_V=V, dec_F=F))
        return V, F

    def dec_hook(feature_dict):
        mean, conc, mix = dec(feature_dict)
        if "mean" not in cap:
            cap.update(dict(mean=mean, concentration=conc, mix_logits=mix))
        return mean, conc, mix

    m.features.features_encode = fe_hook
    m.encode = enc_hook
    m.features.features_decode = fd_hook
    m.decode = dec_hook

    torch.manual_seed(111)
    with torch.no_grad():
        pack_side_chains(fd, m, 1, 16, repack_everything=True)

    cap["periodic_table_features"] = m.features.periodic_table_features
    out = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in cap.items()}
    np.savez(f"{HERE}/packer_capture.npz", **out)
    print("[capture] saved packer_capture.npz with:",
          ", ".join(f"{k}{tuple(v.shape)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
