#!/usr/bin/env python3
"""
Milestone 1 - Export ProteinMPNN's static path (unconditional_probs) to Core ML
and verify on-device numerical parity vs the Milestone-0 oracle.

This is the score-only / per-position-probabilities MVP: no autoregressive loop,
no sampling. It proves the featurization + encoder + decoder ops lower correctly
(topk / gather / einsum / one_hot) before any Swift is written.

Findings baked in from the first run:
  * coremltools' TorchScript frontend trips over `np.minimum(self.top_k, X.shape[1])`
    (dynamic-k topk -> aten::Int on a non-scalar). Fix: pin k to a constant for a
    fixed-length export by patching ProteinFeatures._dist.
  * The torch.export (ExportedProgram) frontend is the modern path; we try it first
    and fall back to torch.jit.trace.

Usage:
  .venv/bin/python milestone1_coreml_export.py --oracle oracle_6MRR.npz \
      --weights /Users/jcastellanos/repos/ProteinMPNN/vanilla_model_weights/v_48_020.pt
"""
import argparse, os, sys, types
import numpy as np
import torch

REPO = "/Users/jcastellanos/repos/ProteinMPNN"
sys.path.insert(0, REPO)
from protein_mpnn_utils import ProteinMPNN  # noqa: E402


class UncondWrapper(torch.nn.Module):
    """Static graph: (X, mask, residue_idx, chain_encoding_all) -> log_probs [B,L,21]."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, mask, residue_idx, chain_encoding_all):
        return self.model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)


def patch_dist_constant_k(model, k):
    """Replace ProteinFeatures._dist with a version using a constant python-int k
    (removes the export-hostile np.minimum(top_k, X.shape[1]) dynamic-k)."""
    feats = model.features

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
        return D_neighbors, E_idx

    feats._dist = types.MethodType(_dist, feats)


def build_model(weights):
    ckpt = torch.load(weights, map_location="cpu")
    m = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    augment_eps=0.0, k_neighbors=ckpt["num_edges"], ca_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval(), int(ckpt["num_edges"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="oracle_6MRR.npz")
    ap.add_argument("--weights", default=f"{REPO}/vanilla_model_weights/v_48_020.pt")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    d = np.load(args.oracle, allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    mask = torch.tensor(d["mask"], dtype=torch.float32)
    residue_idx = torch.tensor(d["residue_idx"], dtype=torch.int32)
    chain_enc = torch.tensor(d["chain_encoding_all"], dtype=torch.int32)
    ref = d["log_p_uncond"]
    B, Lmax = mask.shape

    model, kmax = build_model(args.weights)
    k = min(kmax, Lmax)
    patch_dist_constant_k(model, k)
    wrap = UncondWrapper(model)
    print(f"[export] inputs: X{tuple(X.shape)} L={Lmax}  topk k pinned to {k}")

    ex_inputs = (X, mask, residue_idx.long(), chain_enc.long())
    with torch.no_grad():
        torch_out = wrap(*ex_inputs).numpy()
    print(f"[export] torch wrapper vs oracle max|diff| = {np.abs(torch_out - ref).max():.2e}")

    import coremltools as ct

    inputs = [
        ct.TensorType(name="X", shape=[B, Lmax, 4, 3], dtype=np.float32),
        ct.TensorType(name="mask", shape=[B, Lmax], dtype=np.float32),
        ct.TensorType(name="residue_idx", shape=[B, Lmax], dtype=np.int32),
        ct.TensorType(name="chain_encoding_all", shape=[B, Lmax], dtype=np.int32),
    ]
    prec = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    common = dict(inputs=inputs, convert_to="mlprogram",
                  compute_precision=prec,
                  minimum_deployment_target=ct.target.iOS18)
    print(f"[export] compute_precision = {args.precision}")

    mlmodel = None
    # --- Path A: torch.export (modern frontend) ---
    try:
        ep = torch.export.export(wrap, ex_inputs)
        # coremltools 9 + torch 2.13: exported graph is in TRAINING dialect; lower it
        # to the ATEN dialect (core aten decompositions) before conversion.
        ep = ep.run_decompositions({})
        mlmodel = ct.convert(ep, **common)
        print("[export] converted via torch.export ExportedProgram (+run_decompositions)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[export] torch.export path failed: {type(e).__name__}: {str(e)[:200]}")

    # --- Path B: torch.jit.trace fallback ---
    if mlmodel is None:
        with torch.no_grad():
            traced = torch.jit.trace(wrap, ex_inputs, strict=False)
        mlmodel = ct.convert(traced, **common)
        print("[export] converted via torch.jit.trace")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ProteinMPNN_uncond.mlpackage")
    mlmodel.save(out_path)
    print(f"[export] saved -> {out_path}")

    # --- on-device (Core ML runtime) parity ---
    pred = mlmodel.predict({
        "X": X.numpy(), "mask": mask.numpy(),
        "residue_idx": residue_idx.numpy(), "chain_encoding_all": chain_enc.numpy(),
    })
    cm_out = pred[list(pred.keys())[0]]
    md = np.abs(cm_out - ref)
    m = mask.numpy().astype(bool)
    agree = (cm_out.argmax(-1) == ref.argmax(-1))[m].mean()
    print(f"[export] CoreML vs oracle: max|diff|={md.max():.3e}  mean|diff|={md.mean():.3e}")
    print(f"[export] top-1 amino-acid agreement (fp16 CoreML vs fp32 ref): {agree:.4f}")
    print("[export] PASS (fp16)" if md.max() < 5e-2 else "[export] CHECK: diff > fp16 tol ~5e-2")


if __name__ == "__main__":
    main()
