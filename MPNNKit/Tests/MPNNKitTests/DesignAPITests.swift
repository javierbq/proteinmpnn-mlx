import XCTest
import MLX
@testable import MPNNKit

final class DesignAPITests: XCTestCase {
    // Load a design oracle. Files store without batch dim: decoding_order [L] i32,
    // design_top1 [L] i32, design_logits [L,21] f32. Expand to [1,L] / [1,L,21].
    private func designOracle(_ id: String) throws -> (order: MLXArray, top1: MLXArray, logits: MLXArray) {
        let a = try loadArrays(url: mpnnOracleURL(id, "design"))
        let order  = a["decoding_order"]!.asType(.int32).expandedDimensions(axis: 0)   // [L]    -> [1,L]
        let top1   = a["design_top1"]!.asType(.int32).expandedDimensions(axis: 0)      // [L]    -> [1,L]
        let logits = a["design_logits"]!.asType(.float32).expandedDimensions(axis: 0)  // [L,21] -> [1,L,21]
        return (order, top1, logits)
    }

    func testDesignHoldsFixedPositions() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let L = residues.count
        // A native sequence to hold fixed at a few positions (use all-Ala for a deterministic check).
        let native = [Int](repeating: 0, count: L)   // 0 == 'A'
        var opts = MPNNModel.DesignOptions()
        opts.temperature = 0; opts.seed = 0
        opts.fixedPositions = [0, 5, 10]
        opts.nativeSequence = native
        let r = try model.design(residues, options: opts)
        XCTAssertEqual(r.indices.count, L)
        for p in [0, 5, 10] { XCTAssertEqual(r.indices[p], 0, "fixed position \(p) must stay native (A)") }
    }

    func testDesignBiasForcesArgmax() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let L = residues.count
        let trp = 18   // index of 'W' in ACDEFGHIKLMNPQRSTVWYX
        var bias = Array(repeating: [Float](repeating: 0, count: 21), count: L)
        for i in 0 ..< L { bias[i][trp] = 1000 }        // overwhelming bias toward W
        var opts = MPNNModel.DesignOptions(); opts.temperature = 0; opts.seed = 0; opts.bias = bias
        let r = try model.design(residues, options: opts)
        XCTAssertTrue(r.indices.allSatisfy { $0 == trp }, "huge +bias on W must make every position W")
    }

    func testDesignOmitExcludesAA() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let L = residues.count
        var opts0 = MPNNModel.DesignOptions(); opts0.temperature = 0; opts0.seed = 0
        let free = try model.design(residues, options: opts0)
        // Pick an AA the free design actually used, then omit it everywhere.
        let used = Set(free.indices)
        let victim = used.first { $0 != 20 }!
        var opts = MPNNModel.DesignOptions(); opts.temperature = 0; opts.seed = 0
        opts.omit = Array(repeating: [victim], count: L)
        let r = try model.design(residues, options: opts)
        XCTAssertFalse(r.indices.contains(victim), "omitted AA must not appear")
    }

    func testDesignRejectsBadInput() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        var opts = MPNNModel.DesignOptions(); opts.fixedPositions = [0]   // nativeSequence missing
        XCTAssertThrowsError(try model.design(residues, options: opts)) { err in
            XCTAssertEqual(err as? MPNNModel.MPNNInputError, .nativeSequenceRequired)
        }
    }

    func testDecodeLogitBiasNilMatchesDesignOracle() throws {
        for id in ["6MRR", "5L33"] {
            try skipUnlessAssets(id)
            try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "design").path), "design oracle missing")
            let model = try MPNNModel(packDirectory: mpnnPackURL())
            let residues = try loadResidues(id)
            let L = residues.count
            let inp = model.modelInputs(residues)
            let ora = try designOracle(id)
            let (E, eIdx) = featuresDesignE(model.designWeights, inp.X, inp.mask, inp.Ridx, inp.chainLabels, topK: 32)
            let (hV, hE) = encodeDesign(model.designWeights, E, eIdx, inp.mask)
            let Snative = MLXArray.zeros([1, L]).asType(.int32)
            let chainMask = MLXArray.ones([1, L])
            // logitBias defaults to nil → must reproduce the oracle exactly.
            let (S, logits, _) = decodeSequence(model.designWeights, hV, hE, eIdx, Snative, inp.mask, chainMask,
                                                ora.order, mode: .greedy)
            let top1 = mean((S .== ora.top1).asType(.float32)).item(Float.self)
            let dLogits = max(abs(logits - ora.logits)).item(Float.self)
            XCTAssertEqual(top1, 1.0, "\(id): design top-1 must match oracle exactly")
            XCTAssertLessThan(dLogits, 1e-3, "\(id): logits within 1e-3 of oracle")
        }
    }
}
