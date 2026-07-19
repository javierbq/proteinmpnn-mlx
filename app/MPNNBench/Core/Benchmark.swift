// Benchmark.swift — drive the full design -> repack -> write-PDB pipeline for one protein,
// timing each stage (with eval() barriers so MLX's lazy graph is actually realized) and
// sampling peak memory. Also verifies parity vs the bundled oracle when available.
import Foundation
import MLX

struct BenchResult: Identifiable, Codable {
    var id: String
    var L: Int
    var nChains: Int
    var synthetic: Bool
    var featurizeMs: Double = 0
    var encodeMs: Double = 0
    var designMs: Double = 0
    var repackMs: Double = 0
    var pdbMs: Double = 0
    var totalMs: Double = 0
    var peakMB: Double = 0
    var deltaMB: Double = 0
    var scAtoms: Int = 0
    var designTop1: Double? = nil      // % agreement vs oracle (real proteins)
    var repackRmsd: Double? = nil      // Å vs oracle (real proteins)
    var parityPass: Bool? = nil
    var pdbFile: String? = nil
    var error: String? = nil
}

enum Bench {
    private static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    private static func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b - a) / 1_000_000 }

    static func warmup(_ wD: Weights) {
        // realize a small graph once so first-call kernel compilation isn't charged to a protein
        let x = MLXArray.ones([1, 8, 4, 3])
        let (E, _) = featuresDesignE(wD, x, MLXArray.ones([1, 8]), MLXArray(0 ..< 8).reshaped([1, 8]).asType(.float32),
                                     MLXArray.zeros([1, 8]))
        MLX.eval(E)
    }

    static func run(_ meta: ManifestEntry, wD: Weights, wP: Weights, gc: RepackConstants,
                    names: AtomNames, docsDir: URL) -> BenchResult {
        var r = BenchResult(id: meta.id, L: meta.L, nChains: meta.n_chains, synthetic: meta.synthetic)
        do {
            let inp = try Assets.input(meta)
            let designOra = try? Assets.designOracle(meta.id)
            let order = designOra?.decodingOrder ?? MLXArray(0 ..< meta.L).reshaped([1, meta.L]).asType(.int32)

            MLX.GPU.clearCache()
            MLX.GPU.resetPeakMemory()
            let t0 = now()
            let (E, eIdx) = featuresDesignE(wD, inp.X, inp.mask, inp.Ridx, inp.chainLabels, topK: 32)
            MLX.eval(E, eIdx); let t1 = now()
            let (hV, hE) = encodeDesign(wD, E, eIdx, inp.mask)
            MLX.eval(hV, hE); let t2 = now()
            let (S, _, _) = decodeSequence(wD, hV, hE, eIdx, inp.Snative, inp.mask,
                                           MLXArray.ones([1, meta.L]), order, mode: .greedy)
            MLX.eval(S); let t3 = now()
            let rep = repackFull(wP, gc, Xbb: inp.X, S: S, mask: inp.mask, Ridx: inp.Ridx, chainLabels: inp.chainLabels)
            MLX.eval(rep.atom14, rep.atom14Mask, rep.bFactors); let t4 = now()

            let seqMPNN = S[0].asType(.int32).asArray(Int32.self).map { Int($0) }
            let chainLabels = inp.chainLabels[0].asType(.int32).asArray(Int32.self).map { Int($0) }
            let residueIdx = inp.Ridx[0].asType(.int32).asArray(Int32.self).map { Int($0) }
            let pdb = PDBWriter.write(backbone: inp.X, atom14: rep.atom14, atom14Mask: rep.atom14Mask,
                                     bFactors: rep.bFactors, seqMPNN: seqMPNN, chainLabels: chainLabels,
                                     residueIdx: residueIdx, names: names)
            let t5 = now()
            let snap = MLX.GPU.snapshot()

            r.featurizeMs = ms(t0, t1); r.encodeMs = ms(t1, t2); r.designMs = ms(t2, t3)
            r.repackMs = ms(t3, t4); r.pdbMs = ms(t4, t5); r.totalMs = ms(t0, t5)
            r.peakMB = Double(snap.peakMemory) / (1024 * 1024)        // MLX peak allocation
            r.deltaMB = Double(snap.activeMemory) / (1024 * 1024)     // resident model memory
            r.scAtoms = Int(sum(rep.atom14Mask[0..., 0..., 4...]).item(Float.self))

            // write PDB to Documents
            let url = docsDir.appendingPathComponent("\(meta.id)_designed.pdb")
            try pdb.write(to: url, atomically: true, encoding: .utf8)
            r.pdbFile = url.lastPathComponent

            // parity vs oracle (real proteins)
            if let dOra = designOra {
                let mism = Int(sum((S .!= dOra.top1).asType(.int32)).item(Int32.self))
                r.designTop1 = Double(meta.L - mism) / Double(meta.L) * 100
            }
            if let rOra = try? Assets.repackOracle(meta.id) {
                let scMask = rOra.atom14Mask[0..., 0..., 4...].expandedDimensions(axis: -1)
                let diff = (rep.atom14[0..., 0..., 4...] - rOra.atom14[0..., 0..., 4...]) * scMask
                r.repackRmsd = Double(Foundation.sqrt(sum(square(diff)).item(Float.self) / (sum(scMask).item(Float.self) * 3)))
            }
            if let t = r.designTop1, let rm = r.repackRmsd {
                r.parityPass = (t == 100.0 && rm < 1e-2)
            }
        } catch {
            r.error = "\(error)"
        }
        return r
    }
}
