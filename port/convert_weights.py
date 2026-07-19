#!/usr/bin/env python3
"""Convert the LigandMPNN design + SC-packer .pt checkpoints to fp32 safetensors in
MLX layout, written into the app's bundled-assets dir. Keys == PyTorch state_dict keys,
so the mlx-swift port (which mirrors the Python module names) loads them directly.
"""
import os
import sys
import mlx.core as mx

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from mlx_design import load_design_weights                     # noqa: E402
from mlx_packer import load_weights as load_packer_weights     # noqa: E402

LIGAND = os.path.abspath(os.path.join(HERE, "..", "..", "LigandMPNN", "model_params"))
DESIGN_PT = os.path.join(LIGAND, "ligandmpnn_v_32_010_25.pt")
PACKER_PT = os.path.join(LIGAND, "ligandmpnn_sc_v_32_002_16.pt")
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "MPNNBench", "Resources", "app_assets"))
WDIR = os.path.join(ASSETS, "weights")


def dump(name, weights):
    os.makedirs(WDIR, exist_ok=True)
    path = os.path.join(WDIR, name)
    mx.save_safetensors(path, weights, metadata={"format": "mpnnbench"})
    back = mx.load(path)
    assert set(back.keys()) == set(weights.keys()), f"{name}: key set mismatch on round-trip"
    nparam = sum(int(v.size) for v in weights.values())
    mb = os.path.getsize(path) / 1e6
    # spot-check one tensor round-trips exactly
    k0 = next(iter(weights))
    assert float(mx.abs(back[k0] - weights[k0]).max()) == 0.0
    print(f"{name}: {len(weights)} tensors, {nparam:,} params, {mb:.1f} MB  (round-trip OK)")


def main():
    dump("design.safetensors", load_design_weights(DESIGN_PT))
    dump("packer.safetensors", load_packer_weights(PACKER_PT))
    print("wrote to", WDIR)


if __name__ == "__main__":
    main()
