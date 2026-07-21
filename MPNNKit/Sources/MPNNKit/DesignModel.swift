// DesignModel.swift — LigandMPNN sequence design in MLX, mirroring port/mlx_design.py.
// Protein-only: the ligand context is provably inert (verified max|diff|=0), so the
// featurizer computes only E + E_idx and the encoder zeroes the context internally.
import Foundation
import MLX
import MLXRandom

let ALPHABET = Array("ACDEFGHIKLMNPQRSTVWYX")

// exact ProteinFeaturesLigand RBF pair order; atom indices into [N,Ca,C,O,Cb]
private let DESIGN_PAIRS: [(Int, Int)] = [
    (1, 1), (0, 0), (2, 2), (3, 3), (4, 4),
    (1, 0), (1, 2), (1, 3), (1, 4),
    (0, 2), (0, 3), (0, 4), (4, 2), (4, 3), (3, 2),
    (0, 1), (2, 1), (3, 1), (4, 1),
    (2, 0), (3, 0), (4, 0), (2, 4), (3, 4), (2, 3),
]
private let RBF_LO: Float = 2.0, RBF_HI: Float = 22.0, RBF_N = 16

private func getRBF(_ A: MLXArray, _ B: MLXArray, _ eIdx: MLXArray) -> MLXArray {
    // gather neighbour coords then compute K distances (O(L·K)); avoids the O(L²) matrix.
    let Bnb = gatherNodesG(B, eIdx)                                       // [B,L,K,3]
    let Dn = sqrt(sum(square(A.expandedDimensions(axis: 2) - Bnb), axis: -1) + 1e-6)  // [B,L,K]
    return rbf(Dn, RBF_LO, RBF_HI, RBF_N)
}

private func distEidx(_ Ca: MLXArray, _ mask: MLXArray, _ topK: Int) -> MLXArray {
    let m2 = mask.expandedDimensions(axis: 1) * mask.expandedDimensions(axis: 2)   // [B,L,L]
    let dX = Ca.expandedDimensions(axis: 1) - Ca.expandedDimensions(axis: 2)       // [B,L,L,3]
    let D = m2 * sqrt(sum(square(dX), axis: -1) + 1e-6)
    let Dm = max(D, axis: -1, keepDims: true)
    let Dadj = D + (1.0 - m2) * Dm
    let k = Swift.min(topK, Ca.dim(1))
    return argSort(Dadj, axis: -1)[.ellipsis, 0 ..< k].asType(.int32)
}

private func posEmb(_ w: Weights, _ offset: MLXArray, _ echains: MLXArray, maxRel: Int = 32) -> MLXArray {
    let d = clip(offset + Float(maxRel), min: 0, max: Float(2 * maxRel)) * echains
        + (1.0 - echains) * Float(2 * maxRel + 1)
    return linear(w, "features.embeddings.linear", oneHot(d, 2 * maxRel + 2))
}

// design featurization -> (E, E_idx) only (context is inert for protein-only)
func featuresDesignE(_ w: Weights, _ X: MLXArray, _ mask: MLXArray, _ Ridx: MLXArray,
                     _ chainLabels: MLXArray, topK: Int = 32, eIdxOverride: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let N = X[0..., 0..., 0], Ca = X[0..., 0..., 1], C = X[0..., 0..., 2], O = X[0..., 0..., 3]
    let b = Ca - N, c = C - Ca
    let a = crossProd(b, c)
    let Cb = (-0.58273431) * a + 0.56802827 * b + (-0.54067466) * c + Ca
    let atoms = [N, Ca, C, O, Cb]
    let eIdx = eIdxOverride ?? distEidx(Ca, mask, topK)

    let rbfs = DESIGN_PAIRS.map { getRBF(atoms[$0.0], atoms[$0.1], eIdx) }
    let RBFall = concatenated(rbfs, axis: -1)                             // [B,L,K,400]

    var offset = Ridx.expandedDimensions(axis: 2) - Ridx.expandedDimensions(axis: 1)   // [B,L,L]
    offset = gatherEdges(offset.expandedDimensions(axis: -1).asType(.float32), eIdx)[.ellipsis, 0]
    let dch = (chainLabels.expandedDimensions(axis: 2) - chainLabels.expandedDimensions(axis: 1) .== 0)
        .asType(.float32)
    let Ech = gatherEdges(dch.expandedDimensions(axis: -1), eIdx)[.ellipsis, 0]
    let Epos = posEmb(w, offset, Ech)                                    // [B,L,K,16]
    let E = layerNorm(w, "features.norm_edges",
                      linear(w, "features.edge_embedding", concatenated([Epos, RBFall], axis: -1)))
    return (E, eIdx)
}

// encode (ligand_mpnn branch) with the context zeroed (M=1); faithful to ProteinMPNN.encode
func encodeDesign(_ w: Weights, _ E: MLXArray, _ eIdx: MLXArray, _ mask: MLXArray) -> (MLXArray, MLXArray) {
    let B = mask.dim(0), L = mask.dim(1), C = lastDim(E), M = 1
    var hV = MLXArray.zeros([B, L, C])
    var hE = linear(w, "W_e", E)
    let hEcontext = linear(w, "W_v", MLXArray.zeros([B, L, M, C]))        // W_v(0) = bias

    let ma0 = gatherNodes(mask.expandedDimensions(axis: -1), eIdx)[.ellipsis, 0]
    let maskAttend = mask.expandedDimensions(axis: -1) * ma0
    for i in 0 ..< 3 {
        (hV, hE) = encLayer(w, "encoder_layers.\(i)", hV, hE, eIdx, mask, maskAttend)
    }
    var hVC = linear(w, "W_c", hV)
    let Ym = MLXArray.zeros([B, L, M])
    let YmEdges = Ym.expandedDimensions(axis: 3) * Ym.expandedDimensions(axis: 2)   // [B,L,M,M]
    var Ynodes = linear(w, "W_nodes_y", MLXArray.zeros([B, L, M, C]))
    let Yedges = linear(w, "W_edges_y", MLXArray.zeros([B, L, M, M, C]))
    for i in 0 ..< 2 {
        Ynodes = decLayerJ(w, "y_context_encoder_layers.\(i)", Ynodes, Yedges, maskV: Ym, maskAttend: YmEdges)
        let hEcat = concatenated([hEcontext, Ynodes], axis: -1)
        hVC = decLayer(w, "context_encoder_layers.\(i)", hVC, hEcat, maskV: mask, maskAttend: Ym)
    }
    hVC = linear(w, "V_C", hVC)
    hV = hV + layerNorm(w, "V_C_norm", hVC)
    return (hV, hE)
}

// causal order-backward mask via perm^T @ tri @ perm (no einsum)
private func orderMaskBackward(_ order: MLXArray, _ L: Int) -> MLXArray {
    let perm = oneHot(order, L)                                          // [B,L,L]
    let ii = MLXArray(Int32(0) ..< Int32(L)).reshaped([L, 1])
    let jj = MLXArray(Int32(0) ..< Int32(L)).reshaped([1, L])
    let tri = (ii .> jj).asType(.float32)                               // strictly lower [L,L]
    let perm0 = perm[0]                                                  // [L,L]
    return matmul(matmul(perm0.T, tri), perm0).expandedDimensions(axis: 0)   // [1,L,L]
}

// options for a decode step: greedy (RNG-free parity) or temperature sampling
enum SampleMode { case greedy; case sample(temperature: Float) }

func decodeSequence(_ w: Weights, _ hV: MLXArray, _ hE: MLXArray, _ eIdx: MLXArray,
                    _ Snative: MLXArray, _ mask: MLXArray, _ chainMask: MLXArray,
                    _ order: MLXArray, mode: SampleMode, nDec: Int = 3,
                    logitBias: MLXArray? = nil)
    -> (S: MLXArray, logits: MLXArray, logProbs: MLXArray) {
    let B = hV.dim(0), L = hV.dim(1), C = hV.dim(2)
    let orderInts = order[0].asType(.int32).asArray(Int32.self).map { Int($0) }

    let maskAttend = gatherEdges(orderMaskBackward(order, L).expandedDimensions(axis: -1), eIdx)  // [B,L,K,1]
    let mask1D = mask.reshaped([B, L, 1, 1])
    let maskBw = mask1D * maskAttend
    let maskFw = mask1D * (1.0 - maskAttend)

    let hS = MLXArray.zeros([B, L, C])
    let S = MLXArray.zeros([B, L]).asType(.int32)
    let logitsOut = MLXArray.zeros([B, L, 21])
    let logpOut = MLXArray.zeros([B, L, 21])
    let hVStack: [MLXArray] = [hV] + (0 ..< nDec).map { _ in MLXArray.zeros([B, L, C]) }

    let hEXenc = catNeighborsNodesG(MLXArray.zeros([B, L, C]), hE, eIdx)
    let hEXVenc = catNeighborsNodesG(hV, hEXenc, eIdx)
    let hEXVfw = maskFw * hEXVenc
    let Ws = w["W_s.weight"]!
    let cm = mask * chainMask

    for t in orderInts {
        let eIdxT = eIdx[0..., t ..< (t + 1)]
        let hEt = hE[0..., t ..< (t + 1)]
        let hESt = catNeighborsNodesG(hS, hEt, eIdxT)
        let hEXVt = hEXVfw[0..., t ..< (t + 1)]
        let maskBwT = maskBw[0..., t ..< (t + 1)]
        let maskT = mask[0..., t ..< (t + 1)]
        for l in 0 ..< nDec {
            let hESVdec = catNeighborsNodesG(hVStack[l], hESt, eIdxT)
            let hVt = hVStack[l][0..., t ..< (t + 1)]
            let hESVt = maskBwT * hESVdec + hEXVt
            let out = decLayer(w, "decoder_layers.\(l)", hVt, hESVt, maskV: maskT)
            hVStack[l + 1][0..., t ..< (t + 1)] = out
        }
        let hVtF = hVStack[nDec][0..., t ..< (t + 1)].reshaped([B, C])
        let logits0 = linear(w, "W_out", hVtF)                            // [B,21]
        let logits = logitBias == nil ? logits0
                     : logits0 + logitBias![0..., t ..< (t + 1)].reshaped([B, 21])
        let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
        var St: MLXArray
        switch mode {
        case .greedy:
            St = argMax(logits[0..., 0 ..< 20], axis: -1).asType(.int32)
        case .sample(let temperature):
            let probs20 = softmax(logits[0..., 0 ..< 20] / temperature, axis: -1)
            St = MLXRandom.categorical(log(probs20), axis: -1).asType(.int32)
        }
        let cmT = cm[0..., t].asType(.int32)
        let SnT = Snative[0..., t].asType(.int32)
        St = St * cmT + SnT * (1 - cmT)
        logitsOut[0..., t ..< (t + 1)] = logits.expandedDimensions(axis: 1)
        logpOut[0..., t ..< (t + 1)] = logp.expandedDimensions(axis: 1)
        hS[0..., t ..< (t + 1)] = take(Ws, St, axis: 0).expandedDimensions(axis: 1)
        S[0..., t ..< (t + 1)] = St.reshaped([B, 1])
        // Host-driven loop: realize the running state each step so the lazy graph
        // (and its memory) stays bounded instead of growing to O(L) steps at once.
        MLX.eval([hS, S, logitsOut, logpOut] + hVStack)
    }
    return (S, logitsOut, logpOut)
}
