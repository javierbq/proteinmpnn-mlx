// Geometry.swift — side-chain frame geometry in MLX, mirroring port/mlx_geometry.py
// + the backbone-torsion extraction from port/mlx_repack_full.py.
// Backbone frame (Gram-Schmidt), torsions -> global rigid frames, frames -> atom14.
import MLX

// build a [...,3,3] matrix from 9 scalar arrays (each [...])
private func mat3(_ r00: MLXArray, _ r01: MLXArray, _ r02: MLXArray,
                  _ r10: MLXArray, _ r11: MLXArray, _ r12: MLXArray,
                  _ r20: MLXArray, _ r21: MLXArray, _ r22: MLXArray) -> MLXArray {
    stacked([stacked([r00, r01, r02], axis: -1),
             stacked([r10, r11, r12], axis: -1),
             stacked([r20, r21, r22], axis: -1)], axis: -2)
}

// rotation about x: [[1,0,0],[0,c,-s],[0,s,c]]  (c,s shaped [...])
private func rotX(_ c: MLXArray, _ s: MLXArray) -> MLXArray {
    let z = MLXArray.zeros(like: c), o = MLXArray.ones(like: c)
    return mat3(o, z, z, z, c, -s, z, s, c)
}

private func compose(_ R1: MLXArray, _ t1: MLXArray, _ R2: MLXArray, _ t2: MLXArray) -> (MLXArray, MLXArray) {
    (matmul(R1, R2), matmul(R1, t2.expandedDimensions(axis: -1))[.ellipsis, 0] + t1)
}

// backbone frame from N/CA/C (matches OpenFold make_transform_from_reference). -> (R[...,3,3], t=Ca)
func makeBackboneFrame(_ N: MLXArray, _ Ca: MLXArray, _ C: MLXArray, eps: Float = 1e-9) -> (MLXArray, MLXArray) {
    let trans = -Ca
    var n = N + trans
    let c = C + trans
    let cx = c[.ellipsis, 0], cy = c[.ellipsis, 1], cz = c[.ellipsis, 2]
    let z = MLXArray.zeros(like: cx), o = MLXArray.ones(like: cx)

    let n1 = sqrt(eps + cx * cx + cy * cy)
    let s1 = -cy / n1, c1 = cx / n1
    let C1 = mat3(c1, -s1, z, s1, c1, z, z, z, o)

    let n2 = sqrt(eps + cx * cx + cy * cy + cz * cz)
    let s2 = cz / n2, c2 = sqrt(cx * cx + cy * cy) / n2
    let C2 = mat3(c2, z, s2, z, o, z, -s2, z, c2)

    let cRots = matmul(C2, C1)
    n = matmul(cRots, n.expandedDimensions(axis: -1))[.ellipsis, 0]
    let ny = n[.ellipsis, 1], nz = n[.ellipsis, 2]
    let nn = sqrt(eps + ny * ny + nz * nz)
    let sn = -nz / nn, cn = ny / nn
    let Nr = mat3(o, z, z, z, cn, -sn, z, sn, cn)

    let rots = matmul(Nr, cRots).swappedAxes(-2, -1)
    return (rots, Ca)
}

// alpha [B,N,7,2] (omega,phi,psi,chi1..4 as sin/cos); aatype = S_af2. -> (gR[B,N,8,3,3], gt[B,N,8,3])
func torsionAnglesToFrames(_ bbR: MLXArray, _ bbT: MLXArray, _ alpha: MLXArray,
                           _ aatype: MLXArray, _ rrgdf: MLXArray) -> (MLXArray, MLXArray) {
    let B = aatype.dim(0), N = aatype.dim(1)
    let d = take(rrgdf, aatype, axis: 0)                      // [B,N,8,4,4]
    let dR = d[.ellipsis, 0 ..< 3, 0 ..< 3]                   // [B,N,8,3,3]
    let dt = d[.ellipsis, 0 ..< 3, 3]                         // [B,N,8,3]

    let bbRot = concatenated([MLXArray.zeros([B, N, 1, 1]), MLXArray.ones([B, N, 1, 1])], axis: -1)  // [B,N,1,2]
    let a = concatenated([bbRot, alpha], axis: -2)            // [B,N,8,2]
    let R = rotX(a[.ellipsis, 1], a[.ellipsis, 0])           // [B,N,8,3,3]  (c=a1, s=a0)

    let afR = matmul(dR, R)                                   // [B,N,8,3,3]
    let aft = dt
    let c1R = afR[0..., 0..., 4], c1t = aft[0..., 0..., 4]
    let (c2R, c2t) = compose(c1R, c1t, afR[0..., 0..., 5], aft[0..., 0..., 5])
    let (c3R, c3t) = compose(c2R, c2t, afR[0..., 0..., 6], aft[0..., 0..., 6])
    let (c4R, c4t) = compose(c3R, c3t, afR[0..., 0..., 7], aft[0..., 0..., 7])

    let bbRstack = concatenated([afR[0..., 0..., 0 ..< 5], c2R.expandedDimensions(axis: 2),
                                 c3R.expandedDimensions(axis: 2), c4R.expandedDimensions(axis: 2)], axis: 2)
    let bbTstack = concatenated([aft[0..., 0..., 0 ..< 5], c2t.expandedDimensions(axis: 2),
                                 c3t.expandedDimensions(axis: 2), c4t.expandedDimensions(axis: 2)], axis: 2)

    let gR = matmul(bbR.expandedDimensions(axis: 2), bbRstack)
    let gt = matmul(bbR.expandedDimensions(axis: 2), bbTstack.expandedDimensions(axis: -1))[.ellipsis, 0]
        + bbT.expandedDimensions(axis: 2)
    return (gR, gt)
}

func framesToAtom14(_ gR: MLXArray, _ gt: MLXArray, _ aatype: MLXArray,
                    _ groupIdx: MLXArray, _ atomMask: MLXArray, _ lit: MLXArray) -> MLXArray {
    let grp = take(groupIdx, aatype, axis: 0)                 // [B,N,14] int
    let idxR = broadcast(grp.expandedDimensions(axis: -1).expandedDimensions(axis: -1),
                         to: grp.shape + [3, 3])
    let Rat = takeAlong(gR, idxR, axis: 2)                    // [B,N,14,3,3]
    let idxt = broadcast(grp.expandedDimensions(axis: -1), to: grp.shape + [3])
    let tat = takeAlong(gt, idxt, axis: 2)                    // [B,N,14,3]
    let litp = take(lit, aatype, axis: 0)                     // [B,N,14,3]
    let pos = matmul(Rat, litp.expandedDimensions(axis: -1))[.ellipsis, 0] + tat
    let amask = take(atomMask, aatype, axis: 0)               // [B,N,14]
    return pos * amask.expandedDimensions(axis: -1)
}

// ---- backbone torsion extraction (omega/phi/psi) from N,CA,C,O ----
private func cross3(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let a0 = a[.ellipsis, 0], a1 = a[.ellipsis, 1], a2 = a[.ellipsis, 2]
    let b0 = b[.ellipsis, 0], b1 = b[.ellipsis, 1], b2 = b[.ellipsis, 2]
    return stacked([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis: -1)
}

private func dihedralSinCos(_ a0: MLXArray, _ a1: MLXArray, _ a2: MLXArray, _ a3: MLXArray,
                            eps: Float = 1e-8) -> MLXArray {
    var e0 = a2 - a1
    e0 = e0 / sqrt(sum(e0 * e0, axis: -1, keepDims: true) + eps)
    var e1 = a0 - a2
    e1 = e1 - e0 * sum(e0 * e1, axis: -1, keepDims: true)
    e1 = e1 / sqrt(sum(e1 * e1, axis: -1, keepDims: true) + eps)
    let e2 = cross3(e0, e1)
    let d = a3 - a2
    let sinv = sum(e2 * d, axis: -1)
    let cosv = sum(e1 * d, axis: -1)
    let denom = sqrt(sinv * sinv + cosv * cosv + 1e-8)
    return stacked([sinv / denom, cosv / denom], axis: -1)
}

// X_bb [B,L,4,3] (N,CA,C,O) -> [B,L,3,2] omega/phi/psi sin/cos (psi sign-flipped like OpenFold)
func backboneTorsions(_ Xbb: MLXArray) -> MLXArray {
    let N = Xbb[0..., 0..., 0], CA = Xbb[0..., 0..., 1], C = Xbb[0..., 0..., 2], O = Xbb[0..., 0..., 3]
    let zrow = MLXArray.zeros(like: CA[0..., 0 ..< 1])
    let prevCA = concatenated([zrow, CA[0..., 0 ..< (CA.dim(1) - 1)]], axis: 1)
    let prevC = concatenated([zrow, C[0..., 0 ..< (C.dim(1) - 1)]], axis: 1)
    let omega = dihedralSinCos(prevCA, prevC, N, CA)
    let phi = dihedralSinCos(prevC, N, CA, C)
    let psi = dihedralSinCos(N, CA, C, O) * MLXArray([Float(-1.0), Float(-1.0)])
    return stacked([omega, phi, psi], axis: 2)
}
