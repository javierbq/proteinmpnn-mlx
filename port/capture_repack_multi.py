#!/usr/bin/env python3
"""Capture LigandMPNN packer intermediates + geometry inputs for MULTIPLE proteins
under a FIXED seed, so the MLX port can be validated for reproducibility."""
import os, sys
import numpy as np
import torch

LIG = "/Users/jcastellanos/repos/LigandMPNN"
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIG)
from data_utils import parse_PDB, featurize          # noqa
from sc_utils import Packer, pack_side_chains          # noqa
from openfold.utils import feats                       # noqa

DEVICE = torch.device("cpu")
SC_CKPT = f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt"
SEED = 111
PDBS = {
    "6MRR": f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb",
    "5L33": f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb",
    "4GYT": f"{REPO}/inputs/PDB_homooligomers/pdbs/4GYT.pdb",
    "3HTN": f"{REPO}/inputs/PDB_complexes/pdbs/3HTN.pdb",
}


def build_model():
    ck = torch.load(SC_CKPT, map_location=DEVICE)
    m = Packer(node_features=128, edge_features=128, num_positional_embeddings=16,
               num_chain_embeddings=16, num_rbf=16, hidden_dim=128, num_encoder_layers=3,
               num_decoder_layers=3, atom_context_num=16, lower_bound=0.0, upper_bound=20.0,
               top_k=32, dropout=0.0, augment_eps=0.0, atom37_order=False, device=DEVICE, num_mix=3)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def capture_one(name, pdb, m):
    pd, *_ = parse_PDB(pdb, device=DEVICE, chains=[], parse_all_atoms=True)
    pd["chain_mask"] = torch.ones_like(pd["mask"])
    fd = featurize(pd, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=16, model_type="ligand_mpnn")
    fd["S"] = fd["S"].long()
    cap = {"NCAC": fd["X"][:, :, :3].detach().clone(), "periodic_table_features": m.features.periodic_table_features}

    fe, fdle, enc, dec = m.features.features_encode, m.features.features_decode, m.encode, m.decode
    taf = feats.torsion_angles_to_frames

    def fe_hook(f):
        for k in ("S", "X", "Y", "Y_m", "Y_t", "mask", "R_idx", "chain_labels"):
            cap[f"enc_in_{k}"] = f[k].detach().clone()
        V, E, Ei, Yn, Ye, Ec, Ym = fe(f)
        cap.update(V=V, E=E, E_idx=Ei, Y_nodes=Yn, Y_edges=Ye, E_context=Ec, Y_m=Ym, mask=f["mask"])
        return V, E, Ei, Yn, Ye, Ec, Ym

    def enc_hook(f):
        hV, hE, Ei = enc(f); cap.update(h_V=hV, h_E=hE); return hV, hE, Ei

    def fd_hook(f):
        if "dec_in_X" not in cap:
            for k in ("S", "X", "X_m", "mask", "E_idx", "Y", "Y_m", "Y_t"):
                cap[f"dec_in_{k}"] = f[k].detach().clone()
        V, F = fdle(f)
        if "dec_V" not in cap:
            cap.update(dec_V=V, dec_F=F)
        return V, F

    def dec_hook(f):
        mean, conc, mix = dec(f)
        if "mean" not in cap:
            cap.update(mean=mean, concentration=conc, mix_logits=mix)
        return mean, conc, mix

    def taf_hook(r, alpha, aatype, rrgdf):
        out = taf(r, alpha, aatype, rrgdf)
        if "alpha" not in cap:
            cap.update(alpha=alpha.detach().clone(), aatype=aatype.detach().clone(),
                       bb4x4=r.to_tensor_4x4().detach().clone(), ref_atom14=None)
        return out

    m.features.features_encode = fe_hook
    m.encode = enc_hook
    m.features.features_decode = fd_hook
    m.decode = dec_hook
    feats.torsion_angles_to_frames = taf_hook
    torch.manual_seed(SEED); np.random.seed(SEED)
    with torch.no_grad():
        sc = pack_side_chains(fd, m, 1, 16, repack_everything=True)
    cap["ref_atom14_sampled"] = sc["X"].detach().clone()
    m.features.features_encode, m.encode, m.features.features_decode, m.decode = fe, enc, fdle, dec
    feats.torsion_angles_to_frames = taf

    out = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in cap.items() if v is not None}
    np.savez(f"{HERE}/cap_{name}.npz", **out)
    print(f"[cap] {name}: L={out['mask'].shape[1]}  saved cap_{name}.npz")


def main():
    m = build_model()
    for name, pdb in PDBS.items():
        capture_one(name, pdb, m)


if __name__ == "__main__":
    main()
