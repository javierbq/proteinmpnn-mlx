#!/usr/bin/env python3
"""
Build MPNN.mpnnpack — a small, self-contained, versioned model bundle for import into
RayMol (or any mlx-swift host). Contains ONLY the model (weights + geometry constants +
atom-name tables) and a manifest; NOT the benchmark inputs/oracles.

Files are COPIED verbatim from app_assets (byte-identical safetensors — no mx.load/mx.save
round-trip, which would risk the lazy-mmap zeroing bug). Layout mirrors the Swift `Assets`
loader's directory structure, so the RayMol loader is the existing code pointed at the pack.
"""
import os
import json
import struct
import hashlib
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "MPNNBench", "Resources", "app_assets"))
DIST = os.path.abspath(os.path.join(HERE, "..", "dist"))
PACK = os.path.join(DIST, "MPNN.mpnnpack")

# (source-relative-to-assets, dest-relative-to-pack)
FILES = [
    ("weights/design.safetensors", "weights/design.safetensors"),
    ("weights/packer.safetensors", "weights/packer.safetensors"),
    ("geometry.safetensors", "geometry.safetensors"),
    ("atom14_names.json", "atom14_names.json"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def st_param_count(path):
    """Sum of element counts across a safetensors file, read from its JSON header only."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    total = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        shape = v["shape"]
        p = 1
        for d in shape:
            p *= d
        total += p
    return total


def main():
    if os.path.isdir(PACK):
        shutil.rmtree(PACK)
    os.makedirs(PACK)

    manifest = {
        "format": "mpnnpack",
        "version": 1,
        "min_loader_version": 1,
        "description": "LigandMPNN protein-only sequence design + side-chain repack (MLX).",
        "runtime": {"framework": "mlx-swift", "featurization_dtype": "float32", "top_k": 32},
        "models": {
            "design": {
                "file": "weights/design.safetensors",
                "source_checkpoint": "ligandmpnn_v_32_010_25.pt",
                "k_neighbors": 32, "atom_context_num": 25, "protein_only": True,
            },
            "packer": {
                "file": "weights/packer.safetensors",
                "source_checkpoint": "ligandmpnn_sc_v_32_002_16.pt",
                "k_neighbors": 32, "atom_context_num": 16, "num_denoising_steps": 3,
            },
        },
        "geometry": "geometry.safetensors",
        "atom_names": "atom14_names.json",
        "files": {},
    }

    total_bytes = 0
    for src_rel, dst_rel in FILES:
        src = os.path.join(ASSETS, src_rel)
        dst = os.path.join(PACK, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        total_bytes += size
        entry = {"bytes": size, "sha256": sha256(dst)}
        if dst_rel.endswith(".safetensors"):
            entry["params"] = st_param_count(dst)
        manifest["files"][dst_rel] = entry

    with open(os.path.join(PACK, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # single-file installable form
    zpath = os.path.join(DIST, "MPNN.mpnnpack.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, names in os.walk(PACK):
            for name in names:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(PACK)))

    print(f"pack dir : {PACK}")
    print(f"zip      : {zpath}  ({os.path.getsize(zpath)/1e6:.1f} MB)")
    print(f"contents ({total_bytes/1e6:.1f} MB uncompressed):")
    for rel, e in manifest["files"].items():
        p = f"  {e.get('params'):,} params" if "params" in e else ""
        print(f"  {rel:32s} {e['bytes']/1e6:6.2f} MB  sha256:{e['sha256'][:12]}…{p}")


if __name__ == "__main__":
    main()
