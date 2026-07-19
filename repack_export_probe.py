#!/usr/bin/env python3
"""
Probe: does the LigandMPNN side-chain packer export to Core ML?

Two questions:
  (A) the OpenFold GEOMETRY (torsion angles -> rigid frames -> atom14 coords,
      built on openfold's Rigid class) -- the novel risk.
  (B) the packer DECODE step (MPNN message passing + torsion head) -- same op
      family as the design model, expected to lower.

We capture real inputs from one packer run, build an nn.Module for the geometry,
check torch parity, then attempt torch.export -> coremltools.convert and report
exactly what lowers or fails.

  .venv/bin/python repack_export_probe.py
"""
import os, sys, copy
import numpy as np
import torch

LIG = "/Users/jcastellanos/repos/LigandMPNN"
REPO = "/Users/jcastellanos/repos/ProteinMPNN"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIG)
from data_utils import parse_PDB, featurize                 # noqa
from sc_utils import Packer, pack_side_chains                # noqa
from openfold.utils import feats                             # noqa
from openfold.utils.rigid_utils import Rigid                 # noqa
from openfold.np.residue_constants import (                  # noqa
    restype_rigid_group_default_frame, restype_atom14_to_rigid_group,
    restype_atom14_mask, restype_atom14_rigid_group_positions)

DEVICE = torch.device("cpu")
SC_CKPT = f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt"


class OpenFoldGeom(torch.nn.Module):
    """torsions + backbone frames + aatype -> atom14 coordinates (OpenFold)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("default_frames", torch.tensor(restype_rigid_group_default_frame, dtype=torch.float32))
        self.register_buffer("a14_to_grp", torch.tensor(restype_atom14_to_rigid_group))
        self.register_buffer("a14_mask", torch.tensor(restype_atom14_mask, dtype=torch.float32))
        self.register_buffer("a14_lit", torch.tensor(restype_atom14_rigid_group_positions, dtype=torch.float32))

    def forward(self, bb_frames_4x4, torsions, aatype):
        rigids = Rigid.from_tensor_4x4(bb_frames_4x4)
        pred_frames = feats.torsion_angles_to_frames(rigids, torsions, aatype, self.default_frames)
        xyz14 = feats.frames_and_literature_positions_to_atom14_pos(
            pred_frames, aatype, self.default_frames, self.a14_to_grp, self.a14_mask, self.a14_lit)
        return xyz14


def capture_geom_inputs():
    """Run one packer step, capturing the exact args to torsion_angles_to_frames."""
    pd, *_ = parse_PDB(f"{REPO}/inputs/PDB_monomers/pdbs/6MRR.pdb", device=DEVICE,
                       chains=[], parse_all_atoms=True)
    pd["chain_mask"] = torch.ones_like(pd["mask"])
    ck = torch.load(SC_CKPT, map_location=DEVICE)
    model_sc = Packer(node_features=128, edge_features=128, num_positional_embeddings=16,
                      num_chain_embeddings=16, num_rbf=16, hidden_dim=128,
                      num_encoder_layers=3, num_decoder_layers=3, atom_context_num=16,
                      lower_bound=0.0, upper_bound=20.0, top_k=32, dropout=0.0,
                      augment_eps=0.0, atom37_order=False, device=DEVICE, num_mix=3)
    model_sc.load_state_dict(ck["model_state_dict"]); model_sc.eval()
    fd = featurize(pd, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=16, model_type="ligand_mpnn")
    fd["S"] = fd["S"].long()

    cap = {}
    orig = feats.torsion_angles_to_frames
    def hook(rigids, torsions, aatype, default_frames):
        if "bb" not in cap:
            cap["bb"] = rigids.to_tensor_4x4().detach().clone()
            cap["torsions"] = torsions.detach().clone()
            cap["aatype"] = aatype.detach().clone()
        return orig(rigids, torsions, aatype, default_frames)
    feats.torsion_angles_to_frames = hook
    torch.manual_seed(111)
    with torch.no_grad():
        pack_side_chains(fd, model_sc, 1, 16, repack_everything=True)
    feats.torsion_angles_to_frames = orig
    return cap


def main():
    print("=== capturing real geometry inputs from a packer run ===")
    cap = capture_geom_inputs()
    bb, tors, aa = cap["bb"], cap["torsions"], cap["aatype"]
    print(f"[probe] bb_frames{tuple(bb.shape)} torsions{tuple(tors.shape)} aatype{tuple(aa.shape)}")

    geom = OpenFoldGeom().eval()
    with torch.no_grad():
        ref = geom(bb, tors, aa)
    print(f"[probe] geometry torch forward OK -> atom14 {tuple(ref.shape)}, finite={bool(torch.isfinite(ref).all())}")

    print("\n=== (A) export OpenFold geometry to Core ML ===")
    try:
        ep = torch.export.export(geom, (bb, tors, aa))
        print("[probe] torch.export(geometry): OK")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[probe] torch.export(geometry) FAILED: {type(e).__name__}: {str(e)[:160]}")
        return
    try:
        ep = ep.run_decompositions({})
        print("[probe] run_decompositions: OK")
    except Exception as e:
        print(f"[probe] run_decompositions FAILED: {type(e).__name__}: {str(e)[:160]}")
    import coremltools as ct
    try:
        ml = ct.convert(ep, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                        minimum_deployment_target=ct.target.iOS18)
        ml.save(f"{HERE}/OpenFoldGeom.mlpackage")
        pred = ml.predict({list(ml.input_description)[0]: bb.numpy(),
                           list(ml.input_description)[1]: tors.numpy(),
                           list(ml.input_description)[2]: aa.numpy().astype(np.int32)})
        cm = list(pred.values())[0]
        md = float(np.abs(cm - ref.numpy()).max())
        print(f"[probe] (A) GEOMETRY LOWERS: Core ML atom14 max|diff|={md:.3e}  -> OpenFoldGeom.mlpackage")
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")
        print(f"[probe] (A) GEOMETRY DOES NOT LOWER: {type(e).__name__}: {msg[:220]}")


if __name__ == "__main__":
    main()
