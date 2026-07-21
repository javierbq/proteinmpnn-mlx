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

    public enum ScoreMode: Equatable, Sendable { case conditional, unconditional, leaveOneOut }

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

    public struct DesignOptions {
        /// Sampling temperature; 0 → deterministic greedy (argmax).
        public var temperature: Float = 0.1
        /// Seed for the decoding order (and sampling); nil → nondeterministic.
        public var seed: UInt64? = nil
        /// Positions (0-based) whose identity must be held to `nativeSequence`; required iff non-empty.
        public var fixedPositions: Set<Int> = []
        /// Alphabet indices (length L); required iff `fixedPositions` non-empty.
        public var nativeSequence: [Int]? = nil
        /// L × 21 additive logit bias applied at every decode step.
        public var bias: [[Float]]? = nil
        /// Per-position disallowed alphabet indices (length L).
        public var omit: [Set<Int>]? = nil
        public init() {}
    }

    public struct DesignResult {
        public let sequence: String
        public let indices: [Int]
        public let logits: [[Float]]
    }

    public let manifest: MPNNManifest
    private let designW: Weights
    private let packerW: Weights
    private let geom: RepackConstants
    private let names: AtomNames

    var designWeights: Weights { designW }
    var packerWeights: Weights { packerW }

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

    // MARK: — Repack

    public struct RepackResult {
        public let pdb: String
        public let atomConfidence: [[Float]]   // L × 14 packer log-prob per atom slot
    }

    public func repack(_ residues: [Residue], sequence: [Int]) throws -> RepackResult {
        guard !residues.isEmpty else { throw MPNNInputError.emptyResidues }
        let L = residues.count
        guard sequence.count == L else { throw MPNNInputError.sequenceLengthMismatch(expected: L, got: sequence.count) }
        for a in sequence where a < 0 || a >= 21 { throw MPNNInputError.indexOutOfRange(a) }
        let core = repackCore(residues, indices: sequence)
        MLX.eval(core.atom14, core.atom14Mask, core.bFactors)
        let pdb = PDBWriter.write(backbone: core.X, atom14: core.atom14, atom14Mask: core.atom14Mask,
                                  bFactors: core.bFactors, seqMPNN: sequence,
                                  chainLabels: residues.map { $0.chain }, residueIdx: residues.map { $0.resSeq }, names: names)
        return RepackResult(pdb: pdb, atomConfidence: toRows(core.bFactors[0], rows: L, cols: 14))
    }

    /// Shared repack: returns the raw MLXArrays so both repack() and run()'s PDB path use one code path.
    func repackCore(_ residues: [Residue], indices: [Int])
        -> (X: MLXArray, atom14: MLXArray, atom14Mask: MLXArray, bFactors: MLXArray) {
        let (X, mask, Ridx, chainLabels) = modelInputs(residues)
        let S = MLXArray(indices.map { Int32($0) }, [1, residues.count])
        let r = repackFull(packerW, geom, Xbb: X, S: S, mask: mask, Ridx: Ridx, chainLabels: chainLabels)
        return (X, r.atom14, r.atom14Mask, r.bFactors)
    }

    #if DEBUG
    /// Test-only: expose the raw repacked atom14 for RMSD parity.
    func repackAtom14(_ residues: [Residue], indices: [Int]) -> MLXArray {
        let c = repackCore(residues, indices: indices); MLX.eval(c.atom14); return c.atom14
    }
    #endif

    /// Design a sequence with optional fixed positions, per-position logit bias/omit, and returned logits.
    public func design(_ residues: [Residue], options: DesignOptions = DesignOptions()) throws -> DesignResult {
        guard !residues.isEmpty else { throw MPNNInputError.emptyResidues }
        let L = residues.count
        if !options.fixedPositions.isEmpty {
            guard options.nativeSequence != nil else { throw MPNNInputError.nativeSequenceRequired }
            for p in options.fixedPositions where p < 0 || p >= L { throw MPNNInputError.indexOutOfRange(p) }
        }
        if let nat = options.nativeSequence {
            guard nat.count == L else { throw MPNNInputError.sequenceLengthMismatch(expected: L, got: nat.count) }
            for a in nat where a < 0 || a >= 21 { throw MPNNInputError.indexOutOfRange(a) }
        }
        if let b = options.bias {
            guard b.count == L && b.allSatisfy({ $0.count == 21 }) else {
                throw MPNNInputError.biasShapeMismatch(expected: L, got: b.count)
            }
        }

        if let s = options.seed { MLXRandom.seed(s) }
        let (X, mask, Ridx, chainLabels) = modelInputs(residues)
        var cmArr = [Float](repeating: 1, count: L)
        for p in options.fixedPositions { cmArr[p] = 0 }
        let chainMask = MLXArray(cmArr, [1, L])
        let natInts = options.nativeSequence ?? [Int](repeating: 0, count: L)
        let Snative = MLXArray(natInts.map { Int32($0) }, [1, L]).asType(.int32)
        let logitBias = buildLogitBias(L: L, bias: options.bias, omit: options.omit)

        let order = argSort(MLXRandom.normal([1, L]), axis: -1).asType(.int32)
        let (E, eIdx) = featuresDesignE(designW, X, mask, Ridx, chainLabels, topK: 32)
        let (hV, hE) = encodeDesign(designW, E, eIdx, mask)
        let mode: SampleMode = options.temperature <= 0 ? .greedy : .sample(temperature: options.temperature)
        let (S, logits, _) = decodeSequence(designW, hV, hE, eIdx, Snative, mask, chainMask, order,
                                            mode: mode, logitBias: logitBias)
        MLX.eval(S, logits)
        let idx = S[0].asType(.int32).asArray(Int32.self).map { Int($0) }
        return DesignResult(sequence: String(idx.map { Self.alphabet[$0] }),
                            indices: idx,
                            logits: toRows(logits[0], rows: L, cols: 21))
    }
}
