// RepackLoop.swift — full multi-step side-chain repack in MLX, mirroring port/mlx_repack_full.py.
// backbone + sequence -> S_af2/mask -> backbone torsions -> chi=0 init atom14 -> encode once
// -> {re-featurize decode, mode-chi, geometry} x num_steps -> packed atom14 + b_factors.
// Deterministic (mode, RNG-free), ligand context zeroed (protein-only).
import MLX

private let MPNN_TO_AF2: [Int32] = [0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20]
private let I0_SMALL: [Float] = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
private let I0_LARGE: [Float] = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                                 -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
private let LOG_2PI: Float = 1.8378770664093453

private func evalPoly(_ y: MLXArray, _ coef: [Float]) -> MLXArray {
    var r = MLXArray(coef.last!)
    for c in coef.dropLast().reversed() { r = c + y * r }
    return r
}

private func logI0(_ x: MLXArray) -> MLXArray {
    let y = square(x / 3.75)
    let small = log(evalPoly(y, I0_SMALL))
    let large = x - 0.5 * log(x) + log(evalPoly(3.75 / x, I0_LARGE))
    return which(x .< 3.75, small, large)
}

private func logsumexpLast(_ a: MLXArray) -> MLXArray {
    let m = max(a, axis: -1, keepDims: true)
    return (m + log(sum(exp(a - m), axis: -1, keepDims: true)))[.ellipsis, 0]
}

struct RepackConstants {
    let RRGDF, GRP, AMASK, LIT, mpnnToAf2: MLXArray
    init(_ g: [String: MLXArray]) {
        RRGDF = g["RRGDF"]!
        GRP = g["GRP"]!.asType(.int32)
        AMASK = g["AMASK"]!
        LIT = g["LIT"]!
        mpnnToAf2 = MLXArray(MPNN_TO_AF2)
    }
}

func repackFull(_ w: Weights, _ gc: RepackConstants, Xbb: MLXArray, S: MLXArray, mask: MLXArray,
                Ridx: MLXArray, chainLabels: MLXArray, numSteps: Int = 3)
    -> (atom14: MLXArray, atom14Mask: MLXArray, bFactors: MLXArray) {
    let B = S.dim(0), L = S.dim(1)
    let Saf2 = take(gc.mpnnToAf2, S.asType(.int32), axis: 0)                          // [B,L]
    let xyz14m = take(gc.AMASK, Saf2, axis: 0) * mask.expandedDimensions(axis: -1)    // [B,L,14]
    let bbTors = backboneTorsions(Xbb)                                               // [B,L,3,2]
    let (bbR, bbT) = makeBackboneFrame(Xbb[0..., 0..., 0], Xbb[0..., 0..., 1], Xbb[0..., 0..., 2])

    func atom14FromAlpha(_ alpha: MLXArray) -> MLXArray {
        let (gR, gt) = torsionAnglesToFrames(bbR, bbT, alpha, Saf2, gc.RRGDF)
        return framesToAtom14(gR, gt, Saf2, gc.GRP, gc.AMASK, gc.LIT) * xyz14m.expandedDimensions(axis: -1)
    }

    let chiInit = concatenated([MLXArray.zeros([B, L, 4, 1]), MLXArray.ones([B, L, 4, 1])], axis: -1)
    var atom14 = atom14FromAlpha(concatenated([bbTors, chiInit], axis: 2))            // [B,L,7,2] -> atom14

    // encode once on the idealized atom14 (backbone-only featurization)
    let (V, E, eIdx) = featuresEncodePacker(w, S, atom14, mask, Ridx, chainLabels, topK: 32)
    let (hV, hE) = packerEncode(w, V, E, eIdx, mask)

    var mean = MLXArray.zeros([B, L, 4, 3]), conc = mean, mix = mean, chi = MLXArray.zeros([B, L, 4])
    for _ in 0 ..< numSteps {
        let (dV, dF) = featuresDecodePacker(w, S, atom14, xyz14m, mask, eIdx)
        (mean, conc, mix) = packerDecode(w, dV, dF, hV, hE, eIdx, mask)
        let k = argMax(mix, axis: -1)                                                // [B,L,4]
        chi = takeAlong(mean, k.expandedDimensions(axis: -1), axis: -1)[.ellipsis, 0] // [B,L,4]
        let chiSC = stacked([sin(chi), cos(chi)], axis: -1)                          // [B,L,4,2]
        atom14 = atom14FromAlpha(concatenated([bbTors, chiSC], axis: 2))
    }

    // b_factors: mixture log-prob of the mode angles mapped onto atom14
    let logMix = mix - logsumexpLast(mix).expandedDimensions(axis: -1)
    let vm = conc * cos(chi.expandedDimensions(axis: -1) - mean) - LOG_2PI - logI0(conc)  // [B,L,4,numMix]
    let logProb = logsumexpLast(logMix + vm)                                         // [B,L,4]
    var grp = take(gc.GRP, Saf2, axis: 0)                                            // [B,L,14]
    grp = which(grp .< 4, MLXArray(Int32(4)), grp) - 4
    let bFactors = takeAlong(logProb, grp, axis: -1)                                 // [B,L,14]
    return (atom14, xyz14m, bFactors)
}
