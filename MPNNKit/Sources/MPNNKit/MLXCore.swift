// MLXCore.swift — shared MLX primitives, mirroring port/mlx_packer.py + mlx_features.py.
// Weights are a [String: MLXArray] dict keyed by the PyTorch state_dict names, so the
// transcription from the validated Python is 1:1 (w["encoder_layers.0.W1.weight"] etc.).
import Foundation
import MLX

typealias Weights = [String: MLXArray]

let SQRT2: Float = 1.4142135623730951

@inline(__always) func lastDim(_ a: MLXArray) -> Int { a.shape[a.ndim - 1] }

// ---- primitive ops (match PyTorch semantics exactly) ----
func linear(_ w: Weights, _ name: String, _ x: MLXArray) -> MLXArray {
    let W = w[name + ".weight"]!                 // [out, in]
    var y = matmul(x, W.T)
    if let b = w[name + ".bias"] { y = y + b }
    return y
}

func layerNorm(_ w: Weights, _ name: String, _ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let g = w[name + ".weight"]!, b = w[name + ".bias"]!
    let mu = mean(x, axis: -1, keepDims: true)
    let v = mean(square(x - mu), axis: -1, keepDims: true)
    return (x - mu) / sqrt(v + eps) * g + b
}

// exact erf-form GELU (torch.nn.GELU default) — parity with Python
func geluExact(_ x: MLXArray) -> MLXArray { 0.5 * x * (1.0 + erf(x / SQRT2)) }

// torch.nn.Softplus(beta=1, threshold=20)
func softplusT(_ x: MLXArray, beta: Float = 1.0, threshold: Float = 20.0) -> MLXArray {
    let sx = beta * x
    return which(sx .> threshold, x, log1p(exp(sx)) / beta)
}

func oneHot(_ idx: MLXArray, _ n: Int) -> MLXArray {
    let ar = MLXArray(Int32(0) ..< Int32(n))
    return (idx.asType(.int32).expandedDimensions(axis: -1) .== ar).asType(.float32)
}

// ---- graph gather helpers ----
// nodes[B,N,C], idx[B,N,K] -> [B,N,K,C]   (query dim == node count; matches mlx_packer)
func gatherNodes(_ nodes: MLXArray, _ idx: MLXArray) -> MLXArray {
    let B = nodes.dim(0), N = nodes.dim(1), C = nodes.dim(2), K = lastDim(idx)
    let flat = broadcast(idx.reshaped([B, N * K, 1]), to: [B, N * K, C])
    return takeAlong(nodes, flat, axis: 1).reshaped([B, N, K, C])
}

// general: query dim comes from idx (autoregressive single-position: Nq=1, Nn=L)
func gatherNodesG(_ nodes: MLXArray, _ idx: MLXArray) -> MLXArray {
    let B = nodes.dim(0), C = nodes.dim(2), Nq = idx.dim(1), K = idx.dim(2)
    let flat = broadcast(idx.reshaped([B, Nq * K, 1]), to: [B, Nq * K, C])
    return takeAlong(nodes, flat, axis: 1).reshaped([B, Nq, K, C])
}

// edges[B,N,N,C], idx[B,N,K] -> [B,N,K,C]
func gatherEdges(_ edges: MLXArray, _ idx: MLXArray) -> MLXArray {
    let C = lastDim(edges)
    let nb = broadcast(idx.expandedDimensions(axis: -1), to: idx.shape + [C])
    return takeAlong(edges, nb, axis: 2)
}

func catNeighborsNodes(_ hNodes: MLXArray, _ hNeighbors: MLXArray, _ eIdx: MLXArray) -> MLXArray {
    concatenated([hNeighbors, gatherNodes(hNodes, eIdx)], axis: -1)
}
func catNeighborsNodesG(_ hNodes: MLXArray, _ hNeighbors: MLXArray, _ eIdx: MLXArray) -> MLXArray {
    concatenated([hNeighbors, gatherNodesG(hNodes, eIdx)], axis: -1)
}

func ffn(_ w: Weights, _ name: String, _ h: MLXArray) -> MLXArray {
    linear(w, name + ".W_out", geluExact(linear(w, name + ".W_in", h)))
}

// ---- vector helpers (featurization) ----
func crossProd(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let a0 = a[.ellipsis, 0], a1 = a[.ellipsis, 1], a2 = a[.ellipsis, 2]
    let b0 = b[.ellipsis, 0], b1 = b[.ellipsis, 1], b2 = b[.ellipsis, 2]
    return stacked([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis: -1)
}

func normalizeVec(_ v: MLXArray, eps: Float = 1e-12) -> MLXArray {
    let n = sqrt(sum(v * v, axis: -1, keepDims: true))
    return v / maximum(n, MLXArray(eps))
}

// RBF: D[...] -> [..., numBins]
func rbf(_ D: MLXArray, _ lo: Float, _ hi: Float, _ numBins: Int) -> MLXArray {
    let mu = MLXArray(linspace(lo, hi, numBins))         // [numBins]
    let sigma = (hi - lo) / Float(numBins)
    let d = D.expandedDimensions(axis: -1)
    return exp(-square((d - mu) / sigma))
}

// torch/mlx linspace(lo,hi,n): n points, endpoints inclusive, step=(hi-lo)/(n-1).
private func linspace(_ lo: Float, _ hi: Float, _ n: Int) -> [Float] {
    guard n > 1 else { return [lo] }
    let s = (hi - lo) / Float(n - 1)
    return (0..<n).map { lo + Float($0) * s }
}
