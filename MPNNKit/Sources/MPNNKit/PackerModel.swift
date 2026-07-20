// PackerModel.swift — LigandMPNN side-chain packer featurization + encode/decode in MLX,
// mirroring port/mlx_features.py (features_encode/decode) and port/mlx_packer.py (encode/decode).
// Protein-only: ligand context (Y/Y_m/Y_t) is zeroed. The sequence node embedding V IS used
// (h_V = W_v(V)); the ligand nodes/edges are masked out in encode (verified inert, as for design).
import MLX

private let PK_LO: Float = 0.0, PK_HI: Float = 20.0, PK_NRBF = 16

private func pkGetRBF(_ A: MLXArray, _ B: MLXArray, _ eIdx: MLXArray) -> MLXArray {
    // gather the neighbours' coords and compute only the K distances, instead of a
    // full [B,L,L] distance matrix that then gets gathered — same values, O(L·K) not O(L²).
    let Bnb = gatherNodesG(B, eIdx)                                        // [B,L,K,3]
    let Dn = sqrt(sum(square(A.expandedDimensions(axis: 2) - Bnb), axis: -1) + 1e-6)  // [B,L,K]
    return rbf(Dn, PK_LO, PK_HI, PK_NRBF)
}

private func pkDistEidx(_ Ca: MLXArray, _ mask: MLXArray, _ topK: Int) -> MLXArray {
    let m2 = mask.expandedDimensions(axis: 1) * mask.expandedDimensions(axis: 2)
    let dX = Ca.expandedDimensions(axis: 1) - Ca.expandedDimensions(axis: 2)
    let D = m2 * sqrt(sum(square(dX), axis: -1) + 1e-6)
    let Dm = max(D, axis: -1, keepDims: true)
    let Dadj = D + (1.0 - m2) * Dm
    let k = Swift.min(topK, Ca.dim(1))
    return argSort(Dadj, axis: -1)[.ellipsis, 0 ..< k].asType(.int32)
}

private func pkPosEmb(_ w: Weights, _ offset: MLXArray, _ echains: MLXArray, maxRel: Int = 32) -> MLXArray {
    let d = clip(offset + Float(maxRel), min: 0, max: Float(2 * maxRel)) * echains
        + (1.0 - echains) * Float(2 * maxRel + 1)
    return linear(w, "features.positional_embeddings.linear", oneHot(d, 2 * maxRel + 2))
}

// features_encode -> (V=seq nodes, E=backbone edges, E_idx). Ligand context is inert (zeroed in encode).
func featuresEncodePacker(_ w: Weights, _ S: MLXArray, _ X: MLXArray, _ mask: MLXArray,
                          _ Ridx: MLXArray, _ chainLabels: MLXArray, topK: Int = 32) -> (MLXArray, MLXArray, MLXArray) {
    let N = X[0..., 0..., 0], Ca = X[0..., 0..., 1], C = X[0..., 0..., 2], O = X[0..., 0..., 3]
    let b = Ca - N, c = C - Ca
    let a = crossProd(b, c)
    let Cb = (-0.58273431) * a + 0.56802827 * b + (-0.54067466) * c + Ca
    let atoms = [N, Ca, C, O, Cb]
    let eIdx = pkDistEidx(Ca, mask, topK)

    var rbfs: [MLXArray] = []
    for i in 0 ..< 5 { for j in 0 ..< 5 { rbfs.append(pkGetRBF(atoms[i], atoms[j], eIdx)) } }
    let RBFall = concatenated(rbfs, axis: -1)                       // [B,L,K,400]

    var offset = Ridx.expandedDimensions(axis: 2) - Ridx.expandedDimensions(axis: 1)
    offset = gatherEdges(offset.expandedDimensions(axis: -1).asType(.float32), eIdx)[.ellipsis, 0]
    let dch = (chainLabels.expandedDimensions(axis: 2) - chainLabels.expandedDimensions(axis: 1) .== 0).asType(.float32)
    let Ech = gatherEdges(dch.expandedDimensions(axis: -1), eIdx)[.ellipsis, 0]
    let Epos = pkPosEmb(w, offset, Ech)
    let E = layerNorm(w, "features.enc_norm_edges",
                      linear(w, "features.enc_edge_embedding", concatenated([Epos, RBFall], axis: -1)))
    let V = layerNorm(w, "features.enc_norm_nodes", linear(w, "features.enc_node_embedding", oneHot(S, 21)))
    return (V, E, eIdx)
}

// features_decode -> (dV, dF). Faithful (side-chain 14x14 RBFs + sequence edges);
// ligand-distance node features are masked (Y_m=0) so only the Y_t one-hot constant remains.
func featuresDecodePacker(_ w: Weights, _ S: MLXArray, _ X14: MLXArray, _ X14m: MLXArray,
                          _ mask: MLXArray, _ eIdx: MLXArray, atomContext: Int = 16) -> (MLXArray, MLXArray) {
    let B = X14.dim(0), L = X14.dim(1), K = eIdx.dim(2)
    let Xm = X14m * mask.expandedDimensions(axis: -1)              // [B,L,14]
    let XmG = gatherNodes(Xm, eIdx)                                // [B,L,K,14]

    var rbfSC: [MLXArray] = []
    for i in 0 ..< 14 {
        for j in 0 ..< 14 {
            var r = pkGetRBF(X14[0..., 0..., i], X14[0..., 0..., j], eIdx)   // [B,L,K,16]
            r = r * Xm[0..., 0..., i].expandedDimensions(axis: -1).expandedDimensions(axis: -1)
            r = r * XmG[0..., 0..., 0..., j].expandedDimensions(axis: -1)
            rbfSC.append(r)
        }
    }

    // ligand XY node features: rbf part is masked to 0 (Y_m=0); Y_t=0 -> one_hot(0,120) constant.
    let xyRBF = MLXArray.zeros([B, L, 14, atomContext, PK_NRBF])
    let ytZeros = MLXArray.zeros([B, L, atomContext]).asType(.int32)
    var ytOneHot = oneHot(ytZeros, 120).expandedDimensions(axis: 2)               // [B,L,1,M,120]
    ytOneHot = broadcast(ytOneHot, to: [B, L, 14, atomContext, 120])
    var XY = concatenated([xyRBF, ytOneHot], axis: -1)                            // [B,L,14,M,136]
    XY = linear(w, "features.W_XY_project_down1", XY).reshaped([B, L, -1])
    let dV = layerNorm(w, "features.dec_norm_nodes1", linear(w, "features.dec_node_embedding1", XY))

    let S1 = oneHot(S, 21)                                                        // [B,L,21]
    let S1g = gatherNodes(S1, eIdx)                                               // [B,L,K,21]
    let S1e = broadcast(S1.expandedDimensions(axis: 2), to: [B, L, K, 21])
    let Sfeat = concatenated([S1e, S1g], axis: -1)                               // [B,L,K,42]
    let F = concatenated(rbfSC + [Sfeat], axis: -1)
    let dF = layerNorm(w, "features.dec_norm_edges1", linear(w, "features.dec_edge_embedding1", F))
    return (dV, dF)
}

// packer encode (ProteinMPNN.encode ligand_mpnn branch, packer variant with W_e_context).
func packerEncode(_ w: Weights, _ V: MLXArray, _ E: MLXArray, _ eIdx: MLXArray, _ mask: MLXArray,
                  nEnc: Int = 3, nCtx: Int = 2) -> (MLXArray, MLXArray) {
    let B = mask.dim(0), L = mask.dim(1), C = lastDim(E), M = 1
    let hEcontext = linear(w, "W_e_context", MLXArray.zeros([B, L, M, C]))         // ligand ctx zeroed
    var hV = linear(w, "W_v", V)
    var hE = linear(w, "W_e", E)
    let ma0 = gatherNodes(mask.expandedDimensions(axis: -1), eIdx)[.ellipsis, 0]
    let maskAttend = mask.expandedDimensions(axis: -1) * ma0
    for i in 0 ..< nEnc {
        (hV, hE) = encLayer(w, "encoder_layers.\(i)", hV, hE, eIdx, mask, maskAttend)
    }
    var hVC = linear(w, "W_c", hV)
    let Ym = MLXArray.zeros([B, L, M])
    let YmEdges = Ym.expandedDimensions(axis: 3) * Ym.expandedDimensions(axis: 2)
    var Ynodes = linear(w, "W_nodes_y", MLXArray.zeros([B, L, M, C]))
    let Yedges = linear(w, "W_edges_y", MLXArray.zeros([B, L, M, M, C]))
    for i in 0 ..< nCtx {
        Ynodes = decLayerJ(w, "y_context_encoder_layers.\(i)", Ynodes, Yedges, maskV: Ym, maskAttend: YmEdges)
        let hEcat = concatenated([hEcontext, Ynodes], axis: -1)
        hVC = decLayer(w, "context_encoder_layers.\(i)", hVC, hEcat, maskV: mask, maskAttend: Ym)
    }
    hVC = linear(w, "V_C", hVC)
    hV = hV + layerNorm(w, "V_C_norm", hVC)
    return (hV, hE)
}

// packer decode -> (mean, concentration, mix_logits), each [B,L,4,numMix]
func packerDecode(_ w: Weights, _ dV: MLXArray, _ dF: MLXArray, _ hV0: MLXArray, _ hE: MLXArray,
                  _ eIdx: MLXArray, _ mask: MLXArray, nDec: Int = 3, numMix: Int = 3)
    -> (MLXArray, MLXArray, MLXArray) {
    let hF = linear(w, "W_f", dF)
    let hEF = concatenated([hE, hF], axis: -1)
    let hVsc = linear(w, "W_v_sc", dV)
    var hV = linear(w, "linear_down", concatenated([hV0, hVsc], axis: -1))
    for i in 0 ..< nDec {
        let hEV = catNeighborsNodes(hV, hEF, eIdx)
        hV = decLayer(w, "decoder_layers.\(i)", hV, hEV, maskV: mask)
    }
    let B = hV.dim(0), N = hV.dim(1)
    let t = linear(w, "W_torsions", hV).reshaped([B, N, 4, numMix, 3])
    let mean = t[.ellipsis, 0]
    let conc = 0.1 + softplusT(t[.ellipsis, 1])
    let mix = t[.ellipsis, 2]
    return (mean, conc, mix)
}
