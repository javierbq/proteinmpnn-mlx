// MPNNModel.swift — the public MPNNKit API. Loads a `.mpnnpack` and runs LigandMPNN
// protein-only sequence design + side-chain repack on backbone coordinates the host
// (e.g. RayMol) already has in memory. No MLX types leak through the public surface.
import Foundation
import MLX
import MLXRandom

public struct MPNNModel {

    /// One residue's backbone: N, CA, C, O in Ångström, plus chain index and output residue number.
    public struct Residue {
        public var n: SIMD3<Float>
        public var ca: SIMD3<Float>
        public var c: SIMD3<Float>
        public var o: SIMD3<Float>
        public var chain: Int
        public var resSeq: Int
        public init(n: SIMD3<Float>, ca: SIMD3<Float>, c: SIMD3<Float>, o: SIMD3<Float>, chain: Int, resSeq: Int) {
            self.n = n; self.ca = ca; self.c = c; self.o = o; self.chain = chain; self.resSeq = resSeq
        }
    }

    public struct Options {
        /// Sampling temperature; 0 → deterministic greedy (argmax).
        public var temperature: Float = 0.1
        /// Seed for the decoding order (and sampling); nil → nondeterministic.
        public var seed: UInt64? = nil
        /// Also repack side chains and emit a PDB (else sequence only).
        public var repack: Bool = true
        public init() {}
    }

    public struct Result {
        public let sequence: String     // designed sequence, 1-letter, in input residue order
        public let pdb: String?         // repacked structure (nil if repack == false)
        public let designMs: Double
        public let repackMs: Double
    }

    /// Index → 1-letter AA (index 20 = 'X'). Column order of every [L,21] output.
    public static var alphabet: [Character] { ALPHABET }   // ALPHABET is DesignModel.swift:8

    public enum ScoreMode: Equatable { case conditional, unconditional, leaveOneOut }

    public enum MPNNInputError: Error, Equatable {
        case emptyResidues
        case sequenceLengthMismatch(expected: Int, got: Int)
        case sequenceRequired(ScoreMode)
        case nativeSequenceRequired
        case indexOutOfRange(Int)
        case biasShapeMismatch(expected: Int, got: Int)
    }

    /// Build the model input tensors from residues (extracted verbatim from run(), MPNNModel.swift:62-70).
    func modelInputs(_ residues: [Residue]) -> (X: MLXArray, mask: MLXArray, Ridx: MLXArray, chainLabels: MLXArray) {
        let L = residues.count
        var flat = [Float](); flat.reserveCapacity(L * 12)
        for r in residues { for v in [r.n, r.ca, r.c, r.o] { flat.append(v.x); flat.append(v.y); flat.append(v.z) } }
        let X = MLXArray(flat, [1, L, 4, 3])
        let mask = MLXArray.ones([1, L])
        let Ridx = MLXArray(residues.map { Float($0.resSeq) }, [1, L])
        let chainLabels = MLXArray(residues.map { Float($0.chain) }, [1, L])
        return (X, mask, Ridx, chainLabels)
    }

    /// [1,L,21] additive logit bias from an optional L×21 bias and per-position omit sets. nil if both nil.
    func buildLogitBias(L: Int, bias: [[Float]]?, omit: [Set<Int>]?) -> MLXArray? {
        guard bias != nil || omit != nil else { return nil }
        var flat = [Float](repeating: 0, count: L * 21)
        if let bias = bias { for i in 0 ..< L { for a in 0 ..< 21 { flat[i * 21 + a] += bias[i][a] } } }
        if let omit = omit { for i in 0 ..< Swift.min(L, omit.count) { for a in omit[i] where a >= 0 && a < 21 { flat[i * 21 + a] += -1e9 } } }
        return MLXArray(flat, [1, L, 21])
    }

    /// Row-major [rows,cols] MLXArray → [[Float]].
    func toRows(_ a: MLXArray, rows: Int, cols: Int) -> [[Float]] {
        let flat = a.asType(.float32).asArray(Float.self)
        return (0 ..< rows).map { i in Array(flat[(i * cols) ..< ((i + 1) * cols)]) }
    }

    public let manifest: MPNNManifest
    private let designW: Weights
    private let packerW: Weights
    private let geom: RepackConstants
    private let names: AtomNames

    /// Load from an unzipped `.mpnnpack` directory (RayMol unzips on import).
    public init(packDirectory url: URL, verifyHashes: Bool = true) throws {
        let pack = try MPNNPack(directory: url, verifyHashes: verifyHashes)
        manifest = pack.manifest
        designW = try pack.arrays(pack.modelFile("design"))
        packerW = try pack.arrays(pack.modelFile("packer"))
        geom = RepackConstants(try pack.arrays(pack.manifest.geometry))
        names = try JSONDecoder().decode(AtomNames.self, from: pack.data(pack.manifest.atom_names))
    }

    /// Design a sequence for the given backbone (all positions designed), optionally repacking.
    public func run(_ residues: [Residue], options: Options = Options()) throws -> Result {
        precondition(!residues.isEmpty, "MPNNModel.run: empty residue list")
        let L = residues.count
        if let s = options.seed { MLXRandom.seed(s) }

        var flat = [Float](); flat.reserveCapacity(L * 12)
        for r in residues {
            for v in [r.n, r.ca, r.c, r.o] { flat.append(v.x); flat.append(v.y); flat.append(v.z) }
        }
        let X = MLXArray(flat, [1, L, 4, 3])
        let mask = MLXArray.ones([1, L])
        let chainMask = MLXArray.ones([1, L])
        let Ridx = MLXArray(residues.map { Float($0.resSeq) }, [1, L])
        let chainLabels = MLXArray(residues.map { Float($0.chain) }, [1, L])
        let Snative = MLXArray.zeros([1, L]).asType(.int32)
        let order = argSort(MLXRandom.normal([1, L]), axis: -1).asType(.int32)

        let t0 = DispatchTime.now().uptimeNanoseconds
        let (E, eIdx) = featuresDesignE(designW, X, mask, Ridx, chainLabels, topK: 32)
        let (hV, hE) = encodeDesign(designW, E, eIdx, mask)
        let mode: SampleMode = options.temperature <= 0 ? .greedy : .sample(temperature: options.temperature)
        let (S, _, _) = decodeSequence(designW, hV, hE, eIdx, Snative, mask, chainMask, order, mode: mode)
        MLX.eval(S)
        let t1 = DispatchTime.now().uptimeNanoseconds

        let seqInts = S[0].asType(.int32).asArray(Int32.self).map { Int($0) }
        let sequence = String(seqInts.map { ALPHABET[$0] })

        var pdb: String? = nil
        var repackMs = 0.0
        if options.repack {
            let r = repackFull(packerW, geom, Xbb: X, S: S, mask: mask, Ridx: Ridx, chainLabels: chainLabels)
            MLX.eval(r.atom14, r.atom14Mask, r.bFactors)
            repackMs = Double(DispatchTime.now().uptimeNanoseconds - t1) / 1e6
            pdb = PDBWriter.write(backbone: X, atom14: r.atom14, atom14Mask: r.atom14Mask, bFactors: r.bFactors,
                                  seqMPNN: seqInts, chainLabels: residues.map { $0.chain },
                                  residueIdx: residues.map { $0.resSeq }, names: names)
        }
        return Result(sequence: sequence, pdb: pdb,
                      designMs: Double(t1 - t0) / 1e6, repackMs: repackMs)
    }
}
