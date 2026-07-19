#!/usr/bin/env python3
"""Validate the dependency-free geometry port against OpenFold, across several proteins."""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
LIG = "/Users/jcastellanos/repos/LigandMPNN"
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
sys.path.insert(0, PARENT); sys.path.insert(0, LIG); sys.path.insert(0, HERE)

from data_utils import parse_PDB, featurize                 # noqa
from sc_utils import Packer, pack_side_chains                # noqa
from openfold.utils import feats                             # noqa
from openfold.np.residue_constants import (                  # noqa
    restype_rigid_group_default_frame, restype_atom14_to_rigid_group,
    restype_atom14_mask, restype_atom14_rigid_group_positions)
from geometry_port import (torsion_angles_to_frames_port, frames_to_atom14_port,  # noqa
                           make_backbone_frame_port)

DEVICE = torch.device("cpu")
SC_CKPT = f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt"
PDBS = {
    "6MRR": f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb",
    "5L33": f"{REPO}/inputs/PDB_monomers/pdbs/5L33.pdb",
    "4GYT": f"{REPO}/inputs/PDB_homooligomers/pdbs/4GYT.pdb",
}

# constant tables as torch
RRGDF = torch.tensor(np.asarray(restype_rigid_group_default_frame), dtype=torch.float32)
GRP = torch.tensor(np.asarray(restype_atom14_to_rigid_group), dtype=torch.int64)
AMASK = torch.tensor(np.asarray(restype_atom14_mask), dtype=torch.float32)
LIT = torch.tensor(np.asarray(restype_atom14_rigid_group_positions), dtype=torch.float32)


def capture(pdb):
    """Run one packer step; capture (bb_4x4, torsions, aatype) and the OpenFold atom14 ref."""
    pd, *_ = parse_PDB(pdb, device=DEVICE, chains=[], parse_all_atoms=True)
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

    cap = {"NCAC": fd["X"][:, :, :3].detach().clone()}  # original N,CA,C before packing
    orig_taf = feats.torsion_angles_to_frames
    orig_f14 = feats.frames_and_literature_positions_to_atom14_pos
    def taf_hook(r, alpha, aatype, rrgdf):
        out = orig_taf(r, alpha, aatype, rrgdf)
        if "bb" not in cap:
            cap["bb"] = r.to_tensor_4x4().detach().clone()
            cap["alpha"] = alpha.detach().clone()
            cap["aatype"] = aatype.detach().clone()
        return out
    def f14_hook(r, aatype, *a):
        out = orig_f14(r, aatype, *a)
        if "ref" not in cap:
            cap["ref"] = out.detach().clone()
        return out
    feats.torsion_angles_to_frames = taf_hook
    feats.frames_and_literature_positions_to_atom14_pos = f14_hook
    torch.manual_seed(111)
    with torch.no_grad():
        pack_side_chains(fd, m, 1, 16, repack_everything=True)
    feats.torsion_angles_to_frames = orig_taf
    feats.frames_and_literature_positions_to_atom14_pos = orig_f14
    return cap


def main():
    print(f"{'protein':8s}{'L':>5s} | {'frame R diff':>13s}{'frame t diff':>13s} | "
          f"{'atom14 diff(A)':>15s}{'full-chain(A)':>14s}  result")
    print("-" * 78)
    allok = True
    for name, pdb in PDBS.items():
        cap = capture(pdb)
        bb, alpha, aatype, ref = cap["bb"], cap["alpha"], cap["aatype"], cap["ref"]
        NCAC = cap["NCAC"]
        bb_R, bb_t = bb[..., :3, :3], bb[..., :3, 3]

        # (1) backbone frame from coords vs OpenFold's captured frame
        pR, pt = make_backbone_frame_port(NCAC[:, :, 0], NCAC[:, :, 1], NCAC[:, :, 2])
        dR = float((pR - bb_R).abs().max()); dt = float((pt - bb_t).abs().max())

        # (2) atom14 from OpenFold's own frame (isolates the torsion->atom14 port)
        gR, gt = torsion_angles_to_frames_port(bb_R, bb_t, alpha, aatype, RRGDF)
        pos = frames_to_atom14_port(gR, gt, aatype, GRP, AMASK, LIT)
        d14 = float((pos - ref).abs().max())

        # (3) FULL CHAIN: coords -> my frame -> my geometry -> atom14 vs OpenFold ref
        gR2, gt2 = torsion_angles_to_frames_port(pR, pt, alpha, aatype, RRGDF)
        pos2 = frames_to_atom14_port(gR2, gt2, aatype, GRP, AMASK, LIT)
        dfull = float((pos2 - ref).abs().max())

        ok = max(dR, dt, d14, dfull) < 1e-4 and max(gR.dim(), pos.dim()) <= 5
        allok = allok and ok
        print(f"{name:8s}{aatype.shape[1]:5d} | {dR:13.2e}{dt:13.2e} | "
              f"{d14:15.2e}{dfull:14.2e}  {'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print("[geom] FULL GEOMETRY PORT MATCHES OpenFold (rank<=5, dependency-free)"
          if allok else "[geom] MISMATCH")





if __name__ == "__main__":
    main()
