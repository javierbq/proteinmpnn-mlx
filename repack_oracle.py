#!/usr/bin/env python3
"""
Repacking reference oracle - LigandMPNN side-chain packing (ligandmpnn_sc_v_32_002_16).

Runs the multi-step von-Mises-mixture denoising packer on a structure and dumps,
as ground truth for a native (MLX/Metal) reimplementation:
  - per-denoising-step distribution params: mean, concentration, mix_logits (chi1-4)
  - final atom14 side-chain coordinates (X) + atom mask (X_m)
  - per-atom b-factors (von Mises uncertainty) and final log_prob / sample

Usage:
  .venv/bin/python repack_oracle.py --pdb ../LigandMPNN/inputs/1BC8.pdb
"""
import argparse, os, sys
import numpy as np
import torch

LIG = "/Users/jcastellanos/repos/LigandMPNN"
sys.path.insert(0, LIG)
from data_utils import parse_PDB, featurize          # noqa: E402
from sc_utils import Packer, pack_side_chains          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=f"{LIG}/inputs/1BC8.pdb")
    ap.add_argument("--sc_weights", default=f"{LIG}/model_params/ligandmpnn_sc_v_32_002_16.pt")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cpu"

    model_sc = Packer(
        node_features=128, edge_features=128, num_positional_embeddings=16,
        num_chain_embeddings=16, num_rbf=16, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, atom_context_num=16,
        lower_bound=0.0, upper_bound=20.0, top_k=32, dropout=0.0,
        augment_eps=0.0, atom37_order=False, device=device, num_mix=3)
    ck = torch.load(args.sc_weights, map_location=device)
    model_sc.load_state_dict(ck["model_state_dict"])
    model_sc.to(device).eval()
    print(f"[repack] Packer loaded: {sum(p.numel() for p in model_sc.parameters()):,} params, "
          f"atom_context_num={ck.get('atom_context_num')}")

    protein_dict, *_ = parse_PDB(args.pdb, device=device, chains=[], parse_all_atoms=True)
    protein_dict["chain_mask"] = torch.ones_like(protein_dict["mask"])
    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=16, model_type="ligand_mpnn")
    fd["S"] = fd["S"].long()  # one_hot requires LongTensor
    L = int(fd["mask"].sum().item())
    print(f"[repack] {os.path.basename(args.pdb)}: residues(resolved)={L}, "
          f"denoising_steps={args.steps}, samples/step={args.samples}")

    # per-step hook on decode() -> (mean, concentration, mix_logits)
    steps = []
    orig_decode = model_sc.decode
    def hooked(feat):
        out = orig_decode(feat)
        steps.append(tuple(t.detach().cpu().numpy() for t in out))
        return out
    model_sc.decode = hooked

    with torch.no_grad():
        sc = pack_side_chains(fd, model_sc, args.steps, args.samples, repack_everything=True)

    means = np.stack([s[0] for s in steps])          # [steps, B, L, 4]
    concs = np.stack([s[1] for s in steps])          # [steps, B, L, 4]
    mixes = np.stack([s[2] for s in steps])          # [steps, B, L, 4, num_mix]
    print(f"[repack] captured {len(steps)} denoising steps; "
          f"mean{means.shape} concentration{concs.shape} mix_logits{mixes.shape}")
    print(f"[repack] final atom14 X {tuple(sc['X'].shape)}  b_factors {tuple(sc['b_factors'].shape)}")

    out = args.out or f"{os.path.dirname(os.path.abspath(__file__))}/repack_ref.npz"
    np.savez(out,
             step_mean=means, step_concentration=concs, step_mix_logits=mixes,
             X_atom14=sc["X"].cpu().numpy(), X_m=sc["X_m"].cpu().numpy(),
             b_factors=sc["b_factors"].cpu().numpy(),
             final_sample=sc["sample"].cpu().numpy(),
             final_log_prob=sc["log_prob"].cpu().numpy(),
             S=fd["S"].cpu().numpy())
    print(f"[repack] saved reference -> {out}")


if __name__ == "__main__":
    main()
